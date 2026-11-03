# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Connection spec design also follows Harbor agents/model_connection.py at c55e2439d77decb756dcb8f9e05b96141301f6ce.
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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os

from lexverse.config import load_profile


@dataclass(frozen=True)
class ProviderAccess:
    api_key_envs: tuple[str, ...] = ()
    base_url_envs: tuple[str, ...] = ()
    default_url: str | None = None
    extra_envs: tuple[str, ...] = ()


PROVIDERS = {
    'together': ProviderAccess(('TOGETHERAI_API_KEY', 'TOGETHER_API_KEY')),
    'nvidia': ProviderAccess(('NVIDIA_NIM_API_KEY', 'NVIDIA_API_KEY')),
    'github-copilot': ProviderAccess(('GITHUB_TOKEN',)),
    'vllm': ProviderAccess(('OPENAI_API_KEY',), ('OPENAI_BASE_URL',)),
    'openai': ProviderAccess(('OPENAI_API_KEY',), ('OPENAI_BASE_URL', 'OPENAI_API_BASE'), 'https://api.openai.com/v1', ()),
    'azure': ProviderAccess(('AZURE_API_KEY',), ('AZURE_BASE_URL', 'AZURE_API_BASE'), None, ('AZURE_API_VERSION',)),
    'anthropic': ProviderAccess(('ANTHROPIC_API_KEY',), ('ANTHROPIC_BASE_URL',), 'https://api.anthropic.com', ()),
    'cohere': ProviderAccess(('COHERE_API_KEY',), (), None, ()),
    'replicate': ProviderAccess(('REPLICATE_API_KEY',), (), None, ()),
    'openrouter': ProviderAccess(('OPENROUTER_API_KEY',), ('OPENROUTER_BASE_URL',), 'https://openrouter.ai/api/v1', ()),
    'vercel_ai_gateway': ProviderAccess(('VERCEL_AI_GATEWAY_API_KEY',), (), None, ()),
    'datarobot': ProviderAccess(('DATAROBOT_API_TOKEN',), (), None, ()),
    'vertex_ai': ProviderAccess(('VERTEXAI_PROJECT',), (), None, ()),
    'huggingface': ProviderAccess(('HUGGINGFACE_API_KEY',), (), None, ()),
    'ai21': ProviderAccess(('AI21_API_KEY',), (), None, ()),
    'aleph_alpha': ProviderAccess(('ALEPH_ALPHA_API_KEY',), (), None, ()),
    'baseten': ProviderAccess(('BASETEN_API_KEY',), (), None, ()),
    'nlp_cloud': ProviderAccess(('NLP_CLOUD_API_KEY',), (), None, ()),
    'bedrock': ProviderAccess(('AWS_ACCESS_KEY_ID',), (), None, ('AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_REGION')),
    'sagemaker': ProviderAccess(('AWS_SECRET_ACCESS_KEY',), (), None, ('AWS_ACCESS_KEY_ID', 'AWS_SESSION_TOKEN', 'AWS_REGION')),
    'ollama': ProviderAccess((), ('OLLAMA_BASE_URL', 'OLLAMA_API_BASE'), None, ()),
    'anyscale': ProviderAccess(('ANYSCALE_API_KEY',), (), None, ()),
    'deepinfra': ProviderAccess(('DEEPINFRA_API_KEY',), (), None, ()),
    'featherless_ai': ProviderAccess(('FEATHERLESS_AI_API_KEY',), (), None, ()),
    'google': ProviderAccess(('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENERATIVE_AI_API_KEY'), ('GOOGLE_BASE_URL',), 'https://generativelanguage.googleapis.com', ()),
    'groq': ProviderAccess(('GROQ_API_KEY',), (), None, ()),
    'cerebras': ProviderAccess(('CEREBRAS_API_KEY',), (), None, ()),
    'xai': ProviderAccess(('XAI_API_KEY',), (), None, ()),
    'volcengine': ProviderAccess(('VOLCENGINE_API_KEY',), (), None, ()),
    'codestral': ProviderAccess(('CODESTRAL_API_KEY',), (), None, ()),
    'deepseek': ProviderAccess(('DEEPSEEK_API_KEY',), ('DEEPSEEK_BASE_URL',), 'https://api.deepseek.com', ()),
    'mistral': ProviderAccess(('MISTRAL_API_KEY',), ('MISTRAL_BASE_URL',), 'https://api.mistral.ai/v1', ()),
    'palm': ProviderAccess(('PALM_API_KEY',), (), None, ()),
    'perplexity': ProviderAccess(('PERPLEXITYAI_API_KEY',), (), None, ()),
    'voyage': ProviderAccess(('VOYAGE_API_KEY',), (), None, ()),
    'infinity': ProviderAccess(('INFINITY_API_KEY',), (), None, ()),
    'fireworks_ai': ProviderAccess(('FIREWORKS_AI_API_KEY',), (), None, ()),
    'cloudflare': ProviderAccess(('CLOUDFLARE_API_KEY',), (), None, ()),
    'novita': ProviderAccess(('NOVITA_API_KEY',), (), None, ()),
    'nebius': ProviderAccess(('NEBIUS_API_KEY',), (), None, ()),
    'dashscope': ProviderAccess(('DASHSCOPE_API_KEY',), (), None, ()),
    'moonshot': ProviderAccess(('MOONSHOT_API_KEY',), (), None, ()),
    'kimi': ProviderAccess(('KIMI_API_KEY',), (), None, ()),
    'zai': ProviderAccess(('ZAI_API_KEY',), (), None, ()),
}
_PROVIDER_ALIASES = {"openai_compatible": "openai", "litellm_proxy": "openai", "gemini": "google",
                     "ai21_chat": "ai21", "together_ai": "together", "nvidia_nim": "nvidia",
                     "amazon-bedrock": "bedrock"}


@dataclass(frozen=True)
class ModelConnectionSpec:
    provider: str | None = None
    api_key_envs: tuple[str, ...] = ()
    base_url_envs: tuple[str, ...] = ()
    prefixed_model: bool = True
    strip_api_version: bool = False


@dataclass(frozen=True)
class ResolvedModelConnection:
    model_name: str | None
    provider: str | None
    model: str | None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    configured_base_url: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    url_envs: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        endpoint = self.configured_base_url or self.base_url or "agent-native"
        return hashlib.sha256(endpoint.encode()).hexdigest()


def parse_model_name(model_name: str | None, default_provider: str = "openai") -> tuple[str, str | None]:
    provider, separator, name = (model_name or "").partition("/")
    canonical = _PROVIDER_ALIASES.get(provider, provider)
    if separator and canonical in PROVIDERS:
        return canonical, name
    return default_provider, model_name


def api_key_envs(model_name: str) -> list[str]:
    provider, _ = parse_model_name(model_name)
    return list(PROVIDERS[provider].api_key_envs)


def resolve_model_connection(model_name: str | None, spec: ModelConnectionSpec | None, *,
                             profile: str | None = None, env: Mapping[str, str] | None = None,
                             use_api_credentials: bool = True) -> ResolvedModelConnection:
    """Explicit profiles override declared native env aliases; never mutate os.environ."""
    source = os.environ if env is None else env
    provider, model = parse_model_name(model_name)
    if spec is None:
        return ResolvedModelConnection(model_name, None, model)
    provider = spec.provider or provider
    access = PROVIDERS.get(provider, ProviderAccess())
    key_names = tuple(dict.fromkeys((*spec.api_key_envs, *access.api_key_envs)))
    url_names = tuple(dict.fromkeys((*spec.base_url_envs, *access.base_url_envs)))
    credentials = load_profile(profile) if profile and use_api_credentials else None
    api_key = credentials.api_key if credentials else next((source[k] for k in key_names if source.get(k)), None)
    configured_url = credentials.base_url if credentials else next((source[k] for k in url_names if source.get(k)), None)
    if not use_api_credentials:
        api_key, configured_url = None, None
    base_url = configured_url or (access.default_url if api_key else None)
    if base_url and spec.strip_api_version:
        base_url = base_url.rstrip("/").removesuffix("/v1")
    values = {k: api_key for k in key_names if api_key is not None}
    if not use_api_credentials:
        values = {k: "" for k in (*key_names, *url_names)}
    values.update({k: base_url for k in url_names if base_url is not None})
    values.update({k: source[k] for k in access.extra_envs if source.get(k)})
    native_name = f"{provider}/{model}" if model and spec.prefixed_model else model
    prefix, separator, _ = (model_name or "").partition("/")
    if spec.prefixed_model and spec.provider is None and separator and _PROVIDER_ALIASES.get(prefix, prefix) in PROVIDERS:
        native_name = model_name
    return ResolvedModelConnection(native_name, provider, model, api_key, base_url, configured_url, values, url_names)
