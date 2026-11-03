from langchain_core.tools import tool


@tool
def ask_user(question: str, facts_summary: str | None = None) -> str:
    """Ask the user for information or confirmation of a factual summary."""
    raise RuntimeError("ask_user must be intercepted by human-in-the-loop middleware")
