from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from lexverse.utils.trajectory_utils import format_trajectory_json


def convert_langchain_messages_to_atif(
    messages: list[Any],
    *,
    session_id: str | None = None,
    agent_name: str = "lexverse",
    agent_version: str = "1.0.0",
    model_name: str | None = None,
    usage: dict[str, Any] | None = None,
) -> Trajectory | None:
    """
    Convert a sequence of LangChain/LangGraph messages into a standard ATIF Trajectory.
    """
    steps: list[Step] = []
    step_id = 1

    for msg in messages:
        msg_type = getattr(msg, "type", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        raw_content = (
            getattr(msg, "content", None)
            if not isinstance(msg, dict)
            else msg.get("content")
        )

        if isinstance(raw_content, list):
            text_parts = []
            reasoning_parts = []
            for block in raw_content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        text_parts.append(str(block["text"]))
                    elif btype in ("thinking", "reasoning") and (
                        block.get("thinking") or block.get("text")
                    ):
                        reasoning_parts.append(
                            str(block.get("thinking") or block.get("text"))
                        )
                elif isinstance(block, str):
                    text_parts.append(block)
            text_content = "\n".join(text_parts)
            reasoning_content = "\n".join(reasoning_parts) or None
        else:
            text_content = str(raw_content or "")
            reasoning_content = None

        additional_kwargs = getattr(msg, "additional_kwargs", None) or (
            msg.get("additional_kwargs") if isinstance(msg, dict) else {}
        )
        if not reasoning_content and isinstance(additional_kwargs, dict):
            rc = additional_kwargs.get("reasoning_content")
            if rc:
                reasoning_content = str(rc)

        if msg_type in ("system",):
            steps.append(
                Step(
                    step_id=step_id,
                    source="system",
                    message=text_content,
                )
            )
            step_id += 1

        elif msg_type in ("human", "user"):
            steps.append(
                Step(
                    step_id=step_id,
                    source="user",
                    message=text_content,
                )
            )
            step_id += 1

        elif msg_type in ("ai", "assistant"):
            raw_tool_calls = getattr(msg, "tool_calls", None) or (
                msg.get("tool_calls") if isinstance(msg, dict) else None
            )
            tool_calls: list[ToolCall] | None = None
            if raw_tool_calls:
                tool_calls = []
                for idx, tc in enumerate(raw_tool_calls, 1):
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or f"call_{step_id}_{idx}"
                        tc_name = tc.get("name") or (tc.get("function") or {}).get(
                            "name", "unknown"
                        )
                        tc_args = tc.get("args")
                        if tc_args is None:
                            raw_args = (tc.get("function") or {}).get("arguments", {})
                            if isinstance(raw_args, str):
                                try:
                                    tc_args = json.loads(raw_args)
                                except Exception:
                                    tc_args = {"raw": raw_args}
                            else:
                                tc_args = raw_args
                        if not isinstance(tc_args, dict):
                            tc_args = {"value": tc_args}
                        tool_calls.append(
                            ToolCall(
                                tool_call_id=str(tc_id),
                                function_name=str(tc_name),
                                arguments=tc_args,
                            )
                        )

            usage_meta = getattr(msg, "usage_metadata", None) or (
                msg.get("usage_metadata") if isinstance(msg, dict) else None
            )
            step_metrics: Metrics | None = None
            if isinstance(usage_meta, dict):
                in_tok = usage_meta.get("input_tokens")
                out_tok = usage_meta.get("output_tokens")
                if in_tok or out_tok:
                    step_metrics = Metrics(
                        prompt_tokens=in_tok,
                        completion_tokens=out_tok,
                    )

            steps.append(
                Step(
                    step_id=step_id,
                    source="agent",
                    model_name=model_name,
                    message=text_content or ("(tool use)" if tool_calls else ""),
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    metrics=step_metrics,
                    llm_call_count=1,
                )
            )
            step_id += 1

        elif msg_type == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None) or (
                msg.get("tool_call_id") if isinstance(msg, dict) else None
            )
            obs_result = ObservationResult(
                source_call_id=str(tool_call_id) if tool_call_id else None,
                content=text_content,
            )
            # Attach to the most recent agent step that has matching tool_call_id (or last agent step)
            attached = False
            for prev_step in reversed(steps):
                if prev_step.source != "agent":
                    continue
                if tool_call_id and prev_step.tool_calls:
                    if any(
                        tc.tool_call_id == str(tool_call_id)
                        for tc in prev_step.tool_calls
                    ):
                        if prev_step.observation is None:
                            prev_step.observation = Observation(results=[obs_result])
                        else:
                            prev_step.observation.results.append(obs_result)
                        attached = True
                        break
            if not attached:
                for prev_step in reversed(steps):
                    if prev_step.source == "agent":
                        obs_result.source_call_id = None
                        if prev_step.observation is None:
                            prev_step.observation = Observation(results=[obs_result])
                        else:
                            prev_step.observation.results.append(obs_result)
                        break

    if not steps:
        return None

    total_prompt = (usage or {}).get("input_tokens") or None
    total_completion = (usage or {}).get("output_tokens") or None
    if total_prompt is None:
        s = sum((st.metrics.prompt_tokens or 0) for st in steps if st.metrics)
        total_prompt = s or None
    if total_completion is None:
        s = sum((st.metrics.completion_tokens or 0) for st in steps if st.metrics)
        total_completion = s or None

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name=agent_name,
            version=agent_version,
            model_name=model_name,
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_steps=len(steps),
        ),
    )


def write_atif_trajectory(trajectory: Trajectory, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        format_trajectory_json(trajectory.to_json_dict()),
        encoding="utf-8",
    )
