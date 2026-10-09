from lexverse.config import ConfigError, ModelConfig, load_profile


def create_chat_model(config: ModelConfig, *, context_window_tokens: int | None = None):
    if config.provider != "openai_compatible":
        raise ConfigError(
            "Direct weight backends are not yet supported for agent tool calling; "
            "use an OpenAI-compatible local service"
        )
    parameters = config.generation_parameters
    reserved = {"model", "model_name", "base_url", "openai_api_base", "api_key", "openai_api_key",
                "max_retries", "use_responses_api", "profile"}
    if reserved & parameters.keys():
        raise ConfigError("generation_parameters cannot override model connection or runtime settings")
    if context_window_tokens is not None and (type(context_window_tokens) is not int or context_window_tokens <= 0):
        raise ConfigError("context_window_tokens must be a positive integer")
    output_keys = [key for key in ("max_tokens", "max_completion_tokens") if parameters.get(key) is not None]
    if len(output_keys) > 1:
        raise ConfigError("Specify either max_tokens or max_completion_tokens, not both")
    output = parameters[output_keys[0]] if output_keys else None
    if output is not None and (type(output) is not int or output <= 0):
        raise ConfigError("Maximum output tokens must be a positive integer")
    from langchain_openai import ChatOpenAI

    profile = load_profile(config.profile)
    model = ChatOpenAI(
        model=config.name, base_url=profile.base_url, api_key=profile.api_key,
        use_responses_api=False, max_retries=0, **parameters,
    )
    capacity = context_window_tokens or (model.profile or {}).get("max_input_tokens")
    if not isinstance(capacity, int) or capacity <= 0:
        raise ConfigError("Unknown model context capacity; set context_window_tokens in task configuration")
    if output is not None and output >= capacity:
        raise ConfigError("Maximum output tokens must be smaller than context_window_tokens")
    model.profile = {**(model.profile or {}), "max_input_tokens": capacity - (output or 0)}
    return model
