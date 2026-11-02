"""Task network selection and the endpoints needed by its chosen agent."""

import ipaddress
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lexverse.config import ConfigError


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


def runtime_url(url, environment):
    kind = getattr(environment, "value", environment)
    return docker_url(url) if kind == "docker" else url


def resolve_network_policy(spec, endpoints, *, supports_allowlist=False):
    policy = NetworkPolicy(network_mode=spec.network_mode, allowed_hosts=spec.allowed_hosts)
    if policy.network_mode == "public":
        return policy
    if not supports_allowlist:
        raise ConfigError("This agent does not declare support for network allowlists")
    hosts = [*policy.allowed_hosts, *(endpoint_host(runtime_url(url, spec.environment)) for url in endpoints)]
    hosts.extend(endpoint_host(runtime_url(service.url, spec.environment))
                 for service in spec.selection.mcp_servers if service.url)
    return NetworkPolicy(network_mode="allowlist", allowed_hosts=hosts)
