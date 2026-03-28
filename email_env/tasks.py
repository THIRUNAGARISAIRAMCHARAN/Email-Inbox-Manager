"""
Task registry: three fixed ``TaskConfig`` entries and ``TaskSpec`` builders.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from email_env.models import EmailMessage, TaskSpec


class TaskConfig(BaseModel):
    """Static configuration for one benchmark task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Task id / task_type key.")
    difficulty: str = Field(description="Human-readable difficulty tier.")
    max_steps: int = Field(ge=1, description="Step budget for the episode.")
    desc: str = Field(description="Instruction text shown to the agent.")


TASK_REGISTRY: dict[str, TaskConfig] = {
    "easy_reply": TaskConfig(
        id="easy_reply",
        difficulty="easy",
        max_steps=10,
        desc="Find the urgent email from your boss and send a professional reply.",
    ),
    "medium_triage": TaskConfig(
        id="medium_triage",
        difficulty="medium",
        max_steps=20,
        desc="Label newsletters, reply to 3 emails needing responses, delete spam.",
    ),
    "hard_thread": TaskConfig(
        id="hard_thread",
        difficulty="hard",
        max_steps=30,
        desc="Find the Q4 budget thread and compose a reply referencing prior decisions.",
    ),
}

# Builder: (emails, episode_id) -> TaskSpec
TaskBuilder = Callable[[list[EmailMessage], str], TaskSpec]


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task id: {task_id!r}; known: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]


def build_task_spec(emails: list[EmailMessage], episode_id: str, task_id: str) -> TaskSpec:
    """Materialize a ``TaskSpec`` from registry config and current inbox."""
    cfg = get_task_config(task_id)
    ctx: dict = {"episode_id": episode_id}
    if emails:
        ctx["focus_message_id"] = emails[0].message_id
    ctx["message_ids"] = [e.message_id for e in emails]
    return TaskSpec(
        task_id=cfg.id,
        task_type=cfg.id,
        prompt=cfg.desc,
        gold={},
        context=ctx,
        max_steps=cfg.max_steps,
        grader_name=None,
    )


def get_task_builder(task_id: str) -> TaskBuilder:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task id: {task_id!r}; known: {sorted(TASK_REGISTRY)}")

    def _build(emails: list[EmailMessage], episode_id: str) -> TaskSpec:
        return build_task_spec(emails, episode_id, task_id)

    return _build


def list_task_types() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


__all__ = [
    "TASK_REGISTRY",
    "TaskBuilder",
    "TaskConfig",
    "build_task_spec",
    "get_task_builder",
    "get_task_config",
    "list_task_types",
]
