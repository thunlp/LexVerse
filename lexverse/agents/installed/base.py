# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Modified by the LexVerse Contributors for LexVerse package integration and legal capability support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
import asyncio
from dataclasses import dataclass
from functools import wraps
import os
from pathlib import Path
import re
import shlex
from typing import Any, Callable, Literal, override

from lexverse.agents.base import BaseAgent
from lexverse.environments.base import BaseEnvironment, ExecResult
from lexverse.models.agent.context import AgentContext
from lexverse.utils.env import parse_bool_env_value
from lexverse.utils.templating import render_prompt_template


class NonZeroAgentExitCodeError(RuntimeError):
    """Raised when an installed agent command exits with a non-zero return code."""


class ApiError(NonZeroAgentExitCodeError):
    """Base class for model provider API errors detected in agent output."""


class ApiRateLimitError(ApiError):
    pass


class ApiUsageLimitError(ApiError):
    pass


class ApiInternalServerError(ApiError):
    pass


class ApiOverloadedError(ApiError):
    pass


class ApiConnectionClosedError(ApiError):
    pass


class ApiResponseStalledError(ApiError):
    pass


class OutputTokenExceededError(ApiError):
    pass


class ContextWindowExceededError(ApiError):
    pass


class UnknownApiError(ApiError):
    pass


class ApiProviderResourceNotFoundError(ApiError):
    pass


class AgentSafetyRefusalError(ApiError):
    pass


class AgentAuthenticationError(NonZeroAgentExitCodeError):
    pass


class ModelNotFoundError(NonZeroAgentExitCodeError):
    pass


class NetworkConnectionError(NonZeroAgentExitCodeError):
    pass


@dataclass
class ErrorPattern:
    pattern: str
    exception: type[NonZeroAgentExitCodeError]


@dataclass(frozen=True)
class CliFlag:
    """Declarative specification for a kwarg -> CLI flag mapping on an agent."""

    kwarg: str
    cli: str
    type: Literal["str", "int", "float", "bool", "enum", "str_list"] = "str"
    choices: list[str] | None = None
    default: Any = None
    env_fallback: str | None = None
    format: Callable[[Any], str] | str | None = None

    @property
    def name(self) -> str:
        return self.kwarg


@dataclass(frozen=True)
class EnvVar:
    """Declarative specification for a kwarg -> environment variable mapping."""

    kwarg: str
    env: str
    type: Literal["str", "int", "bool", "enum"] = "str"
    choices: list[str] | None = None
    default: Any = None
    env_fallback: str | None = None
    bool_true: str = "true"
    bool_false: str = "false"

    @property
    def name(self) -> str:
        return self.kwarg


def _coerce_value(
    raw: Any,
    spec_type: str,
    name: str,
    choices: list[str] | None = None,
) -> Any:
    if spec_type == "bool":
        return parse_bool_env_value(raw, name=name)
    elif spec_type == "int":
        if isinstance(raw, bool):
            raise ValueError(f"Invalid value for '{name}': expected int, got bool")
        try:
            return int(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid value for '{name}': cannot convert {raw!r} to int"
            ) from e
    elif spec_type == "float":
        if isinstance(raw, bool):
            raise ValueError(f"Invalid value for '{name}': expected float, got bool")
        try:
            return float(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid value for '{name}': cannot convert {raw!r} to float"
            ) from e
    elif spec_type == "enum":
        val = str(raw)
        if choices and val not in choices:
            raise ValueError(
                f"Invalid value for '{name}': {val!r} not in {choices}"
            )
        return val
    elif spec_type == "str_list":
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [str(raw)]
    else:
        return str(raw)


def with_prompt_template(run_method: Callable) -> Callable:
    """Decorator for BaseInstalledAgent.run() that renders prompt templates and populates context post-run."""

    @wraps(run_method)
    async def wrapper(
        self: "BaseInstalledAgent",
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._prompt_template_path is not None:
            rendered_instruction = render_prompt_template(
                self._prompt_template_path, instruction
            )
        else:
            rendered_instruction = instruction

        try:
            await run_method(self, rendered_instruction, environment, context)
        finally:
            try:
                uid, gid = os.getuid(), os.getgid()
                await environment.exec(
                    f"chmod -R u+rwX /workspace /artifacts /logs/agent /logs/verifier 2>/dev/null || true; "
                    f"chown -hR {uid}:{gid} /workspace /artifacts /logs/agent /logs/verifier 2>/dev/null || true",
                    user="root",
                    timeout_sec=15,
                )
            except Exception:
                pass
            self.populate_context_post_run(context)

    return wrapper


class BaseInstalledAgent(BaseAgent, ABC):
    """
    An agent that is installed and executed inside the sandbox environment via CLI.
    """

    CLI_FLAGS: list[CliFlag] = []
    ENV_VARS: list[EnvVar] = []
    ERROR_PATTERNS: list[ErrorPattern] = []
    DEFAULT_SETUP_TIMEOUT_SEC: int = 300

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        version: str | None = None,
        timeout_sec: float | None = None,
        override_setup_timeout_sec: float | None = None,
        resume: bool = False,
        *args,
        **kwargs,
    ):
        self._resolved_cli_flags: dict[str, Any] = {}
        self._resolved_env_vars: dict[str, str] = {}
        extra_env = kwargs.get("extra_env") or {}

        for flag in self.CLI_FLAGS:
            if flag.name in kwargs:
                raw = kwargs.pop(flag.name)
            elif flag.env_fallback:
                raw = extra_env.get(flag.env_fallback) or os.environ.get(
                    flag.env_fallback
                )
            else:
                raw = None

            if raw is None:
                raw = flag.default

            if raw is not None:
                self._resolved_cli_flags[flag.name] = _coerce_value(
                    raw, flag.type, flag.name, flag.choices
                )

        for ev in self.ENV_VARS:
            if ev.name in kwargs:
                raw = kwargs.pop(ev.name)
            elif ev.env_fallback:
                raw = extra_env.get(ev.env_fallback) or os.environ.get(ev.env_fallback)
            else:
                raw = None

            if raw is None:
                raw = ev.default

            if raw is not None:
                coerced = _coerce_value(raw, ev.type, ev.name)
                if ev.type == "bool":
                    self._resolved_env_vars[ev.env] = (
                        ev.bool_true if coerced else ev.bool_false
                    )
                else:
                    self._resolved_env_vars[ev.env] = str(coerced)

        super().__init__(logs_dir, *args, **kwargs)
        self._resolved_flags = self._resolved_cli_flags
        self._compiled_error_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p.exception)
            for p in self.ERROR_PATTERNS
        ]
        self._prompt_template_path = (
            Path(prompt_template_path) if prompt_template_path is not None else None
        )
        self._version = version
        self._resolved_version: str | None = None
        self._timeout_sec = int(timeout_sec) if timeout_sec is not None else None
        self._setup_timeout_sec = (
            int(override_setup_timeout_sec)
            if override_setup_timeout_sec is not None
            else self.DEFAULT_SETUP_TIMEOUT_SEC
        )
        self._resume = resume
        self._command_index = 0
        self._next_command_dir: Path | None = None
        self._in_setup: bool = False

    def _get_env(self, name: str, default: str | None = None) -> str | None:
        if name in self._extra_env:
            return self._extra_env[name]
        return os.environ.get(name, default)

    def _has_env(self, name: str) -> bool:
        return name in self._extra_env or name in os.environ

    def _get_env_prefixed(self, prefix: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for k, v in os.environ.items():
            if k.startswith(prefix):
                result[k[len(prefix) :]] = v
        for k, v in self._extra_env.items():
            if k.startswith(prefix):
                result[k[len(prefix) :]] = v
        return result

    def resolve_env_vars(self) -> dict[str, str]:
        return dict(self._resolved_env_vars)

    @override
    async def resume(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not getattr(self, "SUPPORTS_RESUME", False):
            return await super().resume(instruction, environment, context)
        prev = self._resume
        self._resume = True
        try:
            await self.run(instruction, environment, context)
        finally:
            self._resume = prev

    def build_cli_flags(self) -> str:
        parts: list[str] = []
        flag_map = {f.name: f for f in self.CLI_FLAGS}
        for name, value in self._resolved_cli_flags.items():
            spec = flag_map[name]
            if spec.format is not None:
                if callable(spec.format):
                    formatted = spec.format(value)
                else:
                    formatted = spec.format.format(value=shlex.quote(str(value)))
                if formatted:
                    parts.append(formatted)
            elif spec.type == "bool":
                if value:
                    parts.append(spec.cli)
            elif spec.type == "str_list":
                for item in value:
                    parts.append(f"{spec.cli} {shlex.quote(item)}")
            else:
                parts.append(f"{spec.cli} {shlex.quote(str(value))}")
        return " ".join(parts)

    def version(self) -> str | None:
        return self._resolved_version or self._version

    def get_version_command(self) -> str | None:
        return None

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    async def resolve_version(self, environment: BaseEnvironment) -> None:
        cmd = self.get_version_command()
        if cmd is None:
            return
        try:
            result = await environment.exec(
                command=cmd, user=self._resolve_agent_user(environment)
            )
            if result.return_code == 0 and result.stdout:
                self._resolved_version = self.parse_version(result.stdout)
                self.logger.debug(
                    f"Resolved {self.name()} version: {self._resolved_version}"
                )
        except Exception as e:
            self.logger.debug(f"Failed to resolve {self.name()} version: {e}")

    @property
    def prompt_template_path(self) -> Path | None:
        return self._prompt_template_path

    async def install(self, environment: BaseEnvironment) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement install()."
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        return None

    def _truncate_output(self, text: str | None, max_len: int = 1000) -> str:
        if not text:
            return "None"
        if len(text) <= max_len:
            return text
        head_len = max_len // 4
        tail_len = max_len - head_len
        omitted = len(text) - head_len - tail_len
        return f"{text[:head_len]} ... [{omitted} chars truncated] ... {text[-tail_len:]}"

    def _classify_exec_error(
        self, command: str, result: Any
    ) -> NonZeroAgentExitCodeError:
        detail = (
            f"Command failed (exit {result.return_code}): {command}\n"
            f"stdout: {self._truncate_output(result.stdout)}\n"
            f"stderr: {self._truncate_output(result.stderr)}"
        )
        output = f"{result.stdout or ''}\n{result.stderr or ''}"
        last_match = None
        for compiled, exception in self._compiled_error_patterns:
            for match in compiled.finditer(output):
                if last_match is None or match.end() > last_match[0]:
                    last_match = (match.end(), compiled, exception)
        if last_match is not None:
            return last_match[2](detail)
        return NonZeroAgentExitCodeError(detail)

    def render_instruction(self, instruction: str) -> str:
        if self._prompt_template_path:
            return render_prompt_template(self._prompt_template_path, instruction)
        return instruction

    def _resolve_agent_user(self, environment: BaseEnvironment) -> str | int | None:
        return environment.default_user

    def _allocate_command_dir(self) -> Path:
        if self._next_command_dir is not None:
            command_dir = self._next_command_dir
            self._next_command_dir = None
            return command_dir

        command_dir = self.logs_dir / f"command-{self._command_index}"
        self._command_index += 1
        command_dir.mkdir(parents=True, exist_ok=True)
        return command_dir

    async def _exec_logged(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        user: str | int | None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        log_output: bool = True,
        check: bool = True,
    ) -> ExecResult:
        effective_env: dict[str, str] | None = None
        if self._extra_env or env:
            effective_env = {**self._extra_env, **(env or {})}

        effective_timeout = (
            timeout_sec
            if timeout_sec is not None
            else (self._setup_timeout_sec if self._in_setup else self._timeout_sec)
        )

        command_dir: Path | None = None
        if log_output:
            command_dir = self._allocate_command_dir()
            (command_dir / "command.txt").write_text(command, encoding="utf-8")

        result = await environment.exec(
            command=command,
            cwd=cwd,
            env=effective_env,
            timeout_sec=effective_timeout,
            user=user,
        )

        if command_dir is not None:
            (command_dir / "return-code.txt").write_text(
                str(result.return_code), encoding="utf-8"
            )
            if result.stdout:
                (command_dir / "stdout.txt").write_text(
                    result.stdout, encoding="utf-8"
                )
            if result.stderr:
                (command_dir / "stderr.txt").write_text(
                    result.stderr, encoding="utf-8"
                )

        if check and result.return_code != 0:
            raise self._classify_exec_error(command, result)

        return result

    async def exec_as_root(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        log_output: bool = True,
        check: bool = True,
    ) -> ExecResult:
        # In LocalEnvironment without root privileges, run as current user
        user = None if environment.type().value == "local" else "root"
        return await self._exec_logged(
            environment,
            command,
            user=user,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
            log_output=log_output,
            check=check,
        )

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        log_output: bool = True,
        check: bool = True,
    ) -> ExecResult:
        return await self._exec_logged(
            environment,
            command,
            user=self._resolve_agent_user(environment),
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
            log_output=log_output,
            check=check,
        )

    async def is_installed(self, environment: BaseEnvironment) -> bool:
        cmd = self.get_version_command()
        if cmd is None:
            return False
        try:
            result = await environment.exec(
                command=cmd, user=self._resolve_agent_user(environment)
            )
            if result.return_code == 0 and result.stdout:
                self._resolved_version = self.parse_version(result.stdout)
                return True
        except Exception as e:
            self.logger.debug(f"Pre-install check for {self.name()} failed: {e}")
        return False

    async def setup(self, environment: BaseEnvironment) -> None:
        root_user = None if environment.type().value == "local" else "root"
        await environment.exec(
            command="[ -d /installed-agent ] || mkdir -p /installed-agent 2>/dev/null || true",
            user=root_user,
        )

        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        if await self.is_installed(environment):
            self.logger.info(
                f"{self.name()} is already installed ({self._resolved_version}), skipping install"
            )
            return

        self._next_command_dir = setup_dir
        self._in_setup = True
        try:
            await asyncio.wait_for(
                self.install(environment),
                timeout=self._setup_timeout_sec,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Agent setup timed out after {self._setup_timeout_sec} seconds"
            )
        finally:
            self._in_setup = False
            self._next_command_dir = None

        await self.resolve_version(environment)
