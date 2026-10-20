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

from __future__ import annotations

import os

PROVIDER_KEYS: dict[str, str | list[str]] = {
    "openai": "OPENAI_API_KEY",
    "litellm_proxy": "OPENAI_API_KEY",
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel_ai_gateway": "VERCEL_AI_GATEWAY_API_KEY",
    "datarobot": "DATAROBOT_API_TOKEN",
    "vertex_ai": "VERTEXAI_PROJECT",
    "huggingface": "HUGGINGFACE_API_KEY",
    "ai21": "AI21_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "aleph_alpha": "ALEPH_ALPHA_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "nlp_cloud": "NLP_CLOUD_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "sagemaker": "AWS_SECRET_ACCESS_KEY",
    "ollama": "OLLAMA_API_BASE",
    "anyscale": "ANYSCALE_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "featherless_ai": "FEATHERLESS_AI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "ai21_chat": "AI21_API_KEY",
    "volcengine": "VOLCENGINE_API_KEY",
    "codestral": "CODESTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "palm": "PALM_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "infinity": "INFINITY_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "cloudflare": "CLOUDFLARE_API_KEY",
    "novita": "NOVITA_API_KEY",
    "nebius": "NEBIUS_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "kimi": "KIMI_API_KEY",
    "zai": "ZAI_API_KEY",
}


def get_api_key_var_names_from_model_name(model_name: str) -> list[str]:
    provider: str | None = None
    if "/" in model_name:
        provider = model_name.split("/", 1)[0].lower()

    if provider and provider in PROVIDER_KEYS:
        var_names = PROVIDER_KEYS[provider]
        return [var_names] if isinstance(var_names, str) else list(var_names)

    # Heuristic fallback by model name prefix
    lower = model_name.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return ["OPENAI_API_KEY"]
    if lower.startswith("claude"):
        return ["ANTHROPIC_API_KEY"]
    if lower.startswith("gemini"):
        return ["GEMINI_API_KEY"]
    if lower.startswith("deepseek"):
        return ["DEEPSEEK_API_KEY"]
    if lower.startswith("qwen"):
        return ["DASHSCOPE_API_KEY" if "DASHSCOPE_API_KEY" in os.environ else "OPENAI_API_KEY"]
    if lower.startswith(("kimi", "moonshot")):
        return ["MOONSHOT_API_KEY"]

    try:
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, llm_provider, _, _ = get_llm_provider(model=model_name)
        if llm_provider in PROVIDER_KEYS:
            var_names = PROVIDER_KEYS[llm_provider]
            return [var_names] if isinstance(var_names, str) else list(var_names)
    except Exception:
        pass

    raise ValueError(f"Unknown model or provider for API key resolution: {model_name}")
