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

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, override
import json
import os
import shlex
import uuid

from lexverse.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from lexverse.agents.installed.node_install import nvm_node_install_snippet
from lexverse.environments.base import BaseEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


class Pi(BaseInstalledAgent):
    SUPPORTS_RESUME: bool = True

    _OUTPUT_FILENAME = "pi.txt"

    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        ),
    ]

    @staticmethod
    @override
    def name() -> str:
        return AgentName.PI.value

    @override
    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; pi --version"

    @override
    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"{nvm_node_install_snippet()} && "
                f"npm install -g @mariozechner/pi-coding-agent{version_spec} && "
                "pi --version"
            ),
        )

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Pi's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"for f in $(find {shlex.quote(self.skills_dir)} -name 'SKILL.md' 2>/dev/null); do "
            f"  d=$(dirname \"$f\"); "
            f"  cp -r \"$d\" $HOME/.agents/skills/ 2>/dev/null || true; "
            f"done"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Return a shell command that writes MCP servers to $HOME/.pi/agent/mcp.json."""
        if not self.mcp_servers:
            return None
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                entry: dict[str, Any] = {
                    "command": server.command,
                    "exposure": "direct",
                    "env": {"PYTHONPATH": "/lexverse-repo"},
                }
                if server.args:
                    entry["args"] = server.args
            else:
                entry = {"url": server.url, "exposure": "direct"}
            servers[server.name] = entry
        content = json.dumps({"mcpServers": servers}, indent=2)
        escaped = shlex.quote(content)
        return f"mkdir -p $HOME/.pi/agent && echo {escaped} > $HOME/.pi/agent/mcp.json"

    def _build_custom_models_command(self, provider: str, model_id: str, environment: BaseEnvironment | None = None) -> str | None:
        """Configure $HOME/.pi/agent/models.json when OPENAI_BASE_URL is provided for a custom OpenAI-compatible server."""
        base_url = self._get_env("OPENAI_BASE_URL") or self._get_env("OPENAI_API_BASE") or self._get_env("LLM_BASE_URL")
        api_key = self._get_env("OPENAI_API_KEY") or self._get_env("LLM_API_KEY") or "local-key"
        if not base_url:
            return None
        if environment is not None and getattr(environment.type(), "value", str(environment.type())) == "docker":
            base_url = base_url.replace("://127.0.0.1:", "://host.docker.internal:").replace("://localhost:", "://host.docker.internal:")
        models_cfg = {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "api": "openai-completions",
                    "apiKey": api_key,
                    "models": [
                        {
                            "id": model_id,
                            "contextWindow": 131072,
                            "maxTokens": 8192,
                        }
                    ],
                }
            }
        }
        escaped = shlex.quote(json.dumps(models_cfg, indent=2))
        return f"mkdir -p $HOME/.pi/agent && echo {escaped} > $HOME/.pi/agent/models.json"

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, _ = self.model_name.split("/", 1)

        env: dict[str, str] = {}
        keys: list[str] = []

        if provider == "amazon-bedrock":
            keys.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif provider == "anthropic":
            keys.extend(["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"])
        elif provider == "github-copilot":
            keys.append("GITHUB_TOKEN")
        elif provider == "google":
            keys.extend(
                [
                    "GEMINI_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_LOCATION",
                    "GOOGLE_GENAI_USE_VERTEXAI",
                    "GOOGLE_API_KEY",
                ]
            )
        elif provider == "groq":
            keys.append("GROQ_API_KEY")
        elif provider == "huggingface":
            keys.append("HF_TOKEN")
        elif provider == "mistral":
            keys.append("MISTRAL_API_KEY")
        elif provider == "openai":
            keys.append("OPENAI_API_KEY")
        elif provider == "openrouter":
            keys.append("OPENROUTER_API_KEY")
        elif provider == "xai":
            keys.append("XAI_API_KEY")

        for key in keys:
            val = self._get_env(key)
            if val:
                env[key] = val

        model_id = self.model_name.split("/", 1)[1]
        model_args = (
            f"--provider {shlex.quote(provider)} --model {shlex.quote(model_id)} "
        )

        cli_flags = self.build_cli_flags()
        if cli_flags:
            cli_flags += " "
        resume_flag = "--continue " if self._resume else ""

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            await self.exec_as_agent(environment, command=mcp_command, env=env)

        models_command = self._build_custom_models_command(provider, model_id, environment)
        if models_command:
            await self.exec_as_agent(environment, command=models_command, env=env)

        await self.exec_as_agent(
            environment,
            command=(
                f"if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"pi --print --mode json --session-dir /logs/agent/pi/sessions "
                f"{resume_flag}"
                f"{model_args}"
                f"{cli_flags}"
                f"{escaped_instruction} "
                f'2>&1 </dev/null | grep -v \'"type":"message_update"\' | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}'
            ),
            env=env,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not output_file.exists():
            return

        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0
        total_cost = 0.0

        for line in output_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "message_end":
                    message = event.get("message") or {}
                    if message.get("role") == "assistant":
                        usage = message.get("usage") or {}
                        total_input_tokens += usage.get("input", 0)
                        total_output_tokens += usage.get("output", 0)
                        total_cache_read_tokens += usage.get("cacheRead", 0)
                        total_cache_write_tokens += usage.get("cacheWrite", 0)
                        cost = usage.get("cost") or {}
                        total_cost += cost.get("total", 0.0)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

        context.n_input_tokens = total_input_tokens + total_cache_read_tokens
        context.n_output_tokens = total_output_tokens
        context.n_cache_tokens = total_cache_read_tokens
        context.cost_usd = total_cost if total_cost > 0 else None

        try:
            traj = self._convert_pi_events_to_atif(output_file)
            if traj is not None:
                (self.logs_dir / "trajectory.json").write_text(
                    json.dumps(traj.to_json_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            self.logger.warning(f"Failed to convert Pi events to ATIF trajectory: {exc}")

    def _convert_pi_events_to_atif(self, output_file: Path) -> Trajectory | None:
        steps: list[Step] = []
        session_id = str(uuid.uuid4())
        step_id = 0
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0

        for raw_line in output_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            ev_type = ev.get("type")
            if ev_type == "session":
                session_id = ev.get("id") or session_id
                continue

            if ev_type == "message_end":
                msg = ev.get("message") or {}
                role = msg.get("role")
                ts_ms = msg.get("timestamp")
                ts_iso = (
                    datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
                    if isinstance(ts_ms, (int, float))
                    else None
                )
                content = msg.get("content")
                if role == "user":
                    text = ""
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "\n".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    step_id += 1
                    steps.append(
                        Step(
                            step_id=step_id,
                            timestamp=ts_iso,
                            source="user",
                            message=text,
                        )
                    )
                elif role == "assistant":
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    tool_calls: list[ToolCall] = []
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text" and block.get("text"):
                                text_parts.append(str(block["text"]))
                            elif btype == "thinking" and block.get("thinking"):
                                reasoning_parts.append(str(block["thinking"]))
                            elif btype == "toolCall":
                                tool_calls.append(
                                    ToolCall(
                                        tool_call_id=str(block.get("id") or uuid.uuid4().hex[:12]),
                                        function_name=str(block.get("name") or ""),
                                        arguments=block.get("arguments")
                                        if isinstance(block.get("arguments"), dict)
                                        else {},
                                    )
                                )
                    elif isinstance(content, str):
                        text_parts.append(content)

                    usage = msg.get("usage") or {}
                    p_tok = int(usage.get("input", 0) or 0) + int(usage.get("cacheRead", 0) or 0)
                    c_tok = int(usage.get("output", 0) or 0)
                    cache_tok = int(usage.get("cacheRead", 0) or 0)
                    cost_val = float((usage.get("cost") or {}).get("total", 0.0) or 0.0)
                    total_prompt += p_tok
                    total_completion += c_tok
                    total_cached += cache_tok
                    total_cost += cost_val

                    step_id += 1
                    steps.append(
                        Step(
                            step_id=step_id,
                            timestamp=ts_iso,
                            source="agent",
                            model_name=self.model_name,
                            message="\n".join(text_parts),
                            reasoning_content="\n".join(reasoning_parts) or None,
                            tool_calls=tool_calls or None,
                            metrics=Metrics(
                                prompt_tokens=p_tok,
                                completion_tokens=c_tok,
                                cached_tokens=cache_tok,
                                cost_usd=cost_val if cost_val > 0 else None,
                            ),
                        )
                    )
                elif role == "toolResult":
                    call_id = msg.get("toolCallId")
                    res_content = msg.get("content")
                    out_text = ""
                    if isinstance(res_content, str):
                        out_text = res_content
                    elif isinstance(res_content, list):
                        out_text = "\n".join(
                            b.get("text", "")
                            for b in res_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    # Attach observation to the most recent agent step containing tool_calls
                    attached = False
                    for prev in reversed(steps):
                        if prev.source == "agent" and prev.tool_calls:
                            obs = prev.observation or Observation(results=[])
                            obs.results.append(
                                ObservationResult(
                                    source_call_id=str(call_id) if call_id else None,
                                    content=out_text,
                                )
                            )
                            prev.observation = obs
                            attached = True
                            break
                    if not attached:
                        step_id += 1
                        steps.append(
                            Step(
                                step_id=step_id,
                                timestamp=ts_iso,
                                source="system",
                                message=out_text,
                            )
                        )

        if not steps:
            return None

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name="pi",
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached,
                total_cost_usd=total_cost if total_cost > 0 else None,
                total_steps=len(steps),
            ),
        )
