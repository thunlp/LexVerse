from lexverse.models.trial.result import EnvironmentResult


async def run(task, provider):
    from lexverse.llms.runtime import LocalProvider
    messages = list(task.input.messages or [{"role": "user", "content": task.input.prompt}])
    options = ({"extra_body": {"lexeval_truncate": True}}
               if task.source.name == "lexeval" and isinstance(provider, LocalProvider) else {})
    response = await provider.generate(list(messages), config=options)
    messages.append({"role": "assistant", "content": response.content})
    return EnvironmentResult(status="completed", answer=response.content, messages=messages,
                             dialog_history=[{"actor": task.participants[0].id, "content": response.content, "metadata": {"model": response.model, "usage": response.usage, "finish_reason": response.finish_reason}}],
                             final_state={"data": {"usage": response.usage}})
