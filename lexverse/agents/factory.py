from pathlib import Path


def build_agent(model, root: Path, *, tools=(), skills=None, checkpointer=None,
                middleware=(), interactive=False, system_prompt="", artifacts_root: Path | None = None,
                skills_root: Path | None = None):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemPermission
    from deepagents.profiles import (
        GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile,
    )
    from pydantic import ValidationError

    provider = model._get_ls_params().get("ls_provider")
    identifier = getattr(model, "model_name", None) or getattr(model, "model", None)
    if not provider:
        raise ValueError("Chat model must identify its provider for single-agent configuration")
    key = f"{provider}:{identifier}" if identifier else provider
    register_harness_profile(key, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"execute", "task", "delete"}),
    ))
    root = root.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Agent filesystem must not contain symbolic links")
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    if artifacts_root is not None or skills_root is not None:
        from deepagents.backends import CompositeBackend
        routes = {}
        if skills_root is not None:
            skills_root = skills_root.resolve()
            if any(path.is_symlink() for path in skills_root.rglob("*")):
                raise ValueError("Skills must not contain symbolic links")
            routes["/skills/"] = FilesystemBackend(root_dir=skills_root, virtual_mode=True)
        if artifacts_root is not None:
            artifacts_root = artifacts_root.resolve()
            if any(path.is_symlink() for path in artifacts_root.rglob("*")):
                raise ValueError("Artifacts must not contain symbolic links")
            routes["/artifacts/"] = FilesystemBackend(root_dir=artifacts_root, virtual_mode=True)
        backend = CompositeBackend(default=backend, routes=routes)
    permissions = [
        FilesystemPermission(operations=["read"], paths=["/inputs", "/inputs/**", "/skills", "/skills/**"]),
        FilesystemPermission(operations=["read", "write"], paths=[
            "/workspace", "/workspace/**", "/artifacts", "/artifacts/**",
            "/large_tool_results", "/large_tool_results/**",
            "/conversation_history", "/conversation_history/**", "/blobs", "/blobs/**",
        ]),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
    ]
    user_tool = next((tool for tool in tools if getattr(tool, "name", None) == "ask_user"), None)

    def valid_question(request):
        try:
            arguments = user_tool.args_schema.model_validate(request.tool_call["args"])
            return bool(arguments.question.strip())
        except ValidationError:
            return False

    return create_deep_agent(
        model=model, backend=backend,
        tools=list(tools), skills=skills, middleware=list(middleware),
        permissions=permissions, checkpointer=checkpointer, system_prompt=system_prompt,
        interrupt_on={"ask_user": {"allowed_decisions": ["respond"], "when": valid_question}}
        if user_tool is not None else None,
    )
