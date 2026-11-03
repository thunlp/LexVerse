from pathlib import Path

from lexverse.models.task.task import VerificationSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput
from lexverse.models.task.config import UserTaskSpec


def user_task(task: UserTaskSpec, *, source: Path) -> LexVerseTask:
    root = source if source.is_dir() else source.parent
    return LexVerseTask(
        id=task.id, source=SourceRef(kind="user", name=task.id, sample_id=task.id,
                                   provenance={"config": str(root / "task.toml")}),
        input=TaskInput(instruction=(root / "instruction.md").read_text(encoding="utf-8")),
        participants=[ParticipantSpec(id="assistant", role="respondent")],
        environment=task.environment.model_dump(mode="json", exclude={"skills", "knowledge", "mcp_servers"}),
        capabilities={"skills": task.environment.skills,
                      "knowledge": [str(path) for path in task.environment.knowledge],
                      "mcp_servers": [server.model_dump(mode="json") for server in task.environment.mcp_servers]},
        outputs={"artifacts": [item.model_dump() for item in task.artifacts]},
        metadata={"agent_timeout_sec": task.agent.timeout_sec},
        verification=VerificationSpec(verifier="sandbox", config=task.verifier.model_dump(mode="json"))
                   if task.verifier is not None else None,
    )
