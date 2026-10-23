"""Task network selection and the endpoints needed by its chosen agent."""

import ipaddress
import hashlib
import os
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lexverse.config import ConfigError, load_profile


def normalize_hosts(hosts):
    result = []
    for value in hosts:
        host = value.strip().lower().rstrip(".")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", host):
                raise ValueError("allowed_hosts must contain exact hostnames or IP addresses, without URLs, ports or wildcards")
        if host not in result:
            result.append(host)
    return result


class NetworkPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    network_mode: Literal["public", "allowlist"] = "public"
    allowed_hosts: list[str] = Field(default_factory=list)

    @field_validator("allowed_hosts")
    @classmethod
    def validate_hosts(cls, value):
        return normalize_hosts(value)


def docker_url(url):
    parts = urlsplit(url)
    if parts.hostname in {"localhost", "127.0.0.1", "::1"}:
        return urlunsplit(parts._replace(netloc="host.docker.internal" + (f":{parts.port}" if parts.port else "")))
    return url


def endpoint_host(url):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ConfigError("Network endpoints must be HTTP(S) URLs without embedded credentials")
    return parts.hostname


def mcp_header_env(server, header):
    return "LEXVERSE_MCP_" + hashlib.sha256((server + ":" + header).encode()).hexdigest()[:16].upper()


def resolve_network_policy(spec, agent_name, resources, extra_env):
    policy = NetworkPolicy(network_mode=spec.network_mode, allowed_hosts=spec.allowed_hosts)
    if policy.network_mode == "public":
        return policy
    hosts = list(policy.allowed_hosts)
    if agent_name not in {"oracle", "nop"}:
        if agent_name == "lexverse":
            url = load_profile(spec.model.profile).base_url or "https://api.openai.com/v1"
        elif agent_name == "codex":
            url = extra_env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            if os.environ.get("CODEX_FORCE_AUTH_JSON") or os.environ.get("CODEX_AUTH_JSON_PATH"):
                hosts.extend(["chatgpt.com", "auth.openai.com"])
        elif agent_name == "claude-code":
            url = extra_env.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            if os.environ.get("CLAUDE_FORCE_OAUTH") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
                hosts.append("platform.claude.com")
        else:
            raise ConfigError("allowlist currently supports lexverse, codex, claude-code and oracle; other adapters must declare their model endpoints and disable native web tools first")
        hosts.append(endpoint_host(docker_url(url)))
    registry = resources.get("registry", {}).get("mcp", {})
    for alias in spec.selection.mcp:
        if alias not in registry:
            raise ConfigError(f"Unknown MCP service: {alias}")
        hosts.append(endpoint_host(docker_url(registry[alias]["url"])))
    return NetworkPolicy(network_mode="allowlist", allowed_hosts=hosts)
