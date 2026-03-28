"""
Pydantic v2 models for Email Inbox Management (OpenEnv).

Covers message payloads, environment state, agent observations, step outcomes,
and the discriminated action union the HTTP API accepts on ``POST /step``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums & literals (domain vocabulary)
# ---------------------------------------------------------------------------


class MessageImportance(StrEnum):
    low = "low"
    normal = "normal"
    high = "high"


class TaskId(StrEnum):
    """Canonical task identifiers (match ``TaskSpec.task_id`` / registry)."""

    easy_reply = "easy_reply"
    medium_triage = "medium_triage"
    hard_thread = "hard_thread"


class TaskType(StrEnum):
    """Registered task types (same values as ``TaskId`` in this environment)."""

    easy_reply = "easy_reply"
    medium_triage = "medium_triage"
    hard_thread = "hard_thread"


# Folders referenced by ``medium_triage`` and synthetic data.
StandardFolder: TypeAlias = Literal["inbox", "updates", "finance", "social"]

STANDARD_FOLDERS: tuple[str, ...] = ("inbox", "updates", "finance", "social")

ACTION_TYPE_LITERALS: tuple[str, ...] = (
    "read_email",
    "reply",
    "compose",
    "label",
    "delete",
    "search",
    "noop",
)


# ---------------------------------------------------------------------------
# Core message & task definitions
# ---------------------------------------------------------------------------


class EmailMessage(BaseModel):
    """
    One email in the simulated inbox.

    ``metadata`` holds synthetic grading hints (topic, priority, folder) and
    may be extended by ``data_gen`` or future import pipelines.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    message_id: str = Field(description="Stable unique id for this message.")
    subject: str = Field(min_length=1, description="Subject line as shown in the client.")
    body: str = Field(description="Plain-text body.")
    from_address: str = Field(description="RFC5322-style From address (simulated).")
    to_addresses: list[str] = Field(
        default_factory=list,
        description="Primary recipients.",
    )
    cc_addresses: list[str] = Field(default_factory=list, description="Carbon copy recipients.")
    bcc_addresses: list[str] = Field(default_factory=list, description="Blind carbon copy recipients.")
    date_sent: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was sent (UTC).",
    )
    read: bool = Field(default=False, description="Whether the user has opened/read the message.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation/thread id; None if not part of a modeled thread.",
    )
    in_reply_to: str | None = Field(
        default=None,
        description="``message_id`` of the parent message, if this is a reply.",
    )
    folder: StandardFolder | str = Field(
        default="inbox",
        description="Current folder/label bucket for triage tasks.",
    )
    importance: MessageImportance = Field(
        default=MessageImportance.normal,
        description="Sender/system importance hint.",
    )
    has_attachments: bool = Field(default=False, description="Whether attachments exist (metadata only in scaffold).")
    snippet: str | None = Field(
        default=None,
        description="Short preview text; defaults may be derived from body in generators.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured hints (synthetic topic, gold labels, etc.).",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_snippet_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            body = data.get("body")
            if data.get("snippet") is None and body:
                text = str(body).replace("\n", " ").strip()
                data = {**data, "snippet": text[:200]}
        return data


Email: TypeAlias = EmailMessage


class TaskSpec(BaseModel):
    """
    Episode task: natural-language instruction plus private ``gold`` for graders.

    ``context`` is safe to expose to the agent; ``gold`` should not be echoed in observations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: TaskId | Literal["easy_reply", "medium_triage", "hard_thread"] = Field(
        description="Stable task instance id (here, same as task family id).",
    )
    task_type: TaskType | Literal["easy_reply", "medium_triage", "hard_thread"] = Field(
        description="Which registered task builder produced this spec.",
    )
    prompt: str = Field(description="Instructions shown to the agent.")
    gold: dict[str, Any] = Field(
        default_factory=dict,
        description="Private rubric payload consumed by graders only.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-sensitive structured context (e.g. focus message id).",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Optional step budget for this task; None means environment default.",
    )
    grader_name: str | None = Field(
        default=None,
        description="Optional grader override; None selects environment default mapping.",
    )


# ---------------------------------------------------------------------------
# Environment state (full serializable snapshot)
# ---------------------------------------------------------------------------


class DraftMessage(BaseModel):
    """In-progress compose buffer (not yet sent)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft_id: str = Field(description="Stable id for this draft.")
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    subject: str = ""
    body: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InboxState(BaseModel):
    """
    Complete environment state returned by ``GET /state/{session_id}``.

    The scaffold populates a subset; optional collections default empty until
    the environment implements moves, trash, search results, etc.
    """

    model_config = ConfigDict(extra="forbid")

    session_episode_id: str = Field(description="Unique id for this episode after last reset.")
    task_type: str = Field(description="Task family selected at reset.")
    emails: list[EmailMessage] = Field(description="All messages currently in the primary mailbox view.")
    focus_message_id: str | None = Field(
        default=None,
        description="Message id the task draws attention to, if any.",
    )
    current_task: TaskSpec | None = Field(default=None, description="Active task specification.")
    step_count: int = Field(default=0, ge=0, description="Number of completed step() calls this episode.")
    done: bool = Field(default=False, description="Whether the episode is terminal.")
    last_reward: float | None = Field(
        default=None,
        description="Most recent terminal reward in [0, 1], if graded.",
    )
    trash_message_ids: list[str] = Field(
        default_factory=list,
        description="Ids of messages in trash (soft-delete scaffold).",
    )
    folder_index: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map folder name -> message ids currently assigned.",
    )
    drafts: list[DraftMessage] = Field(default_factory=list, description="Unsent compose drafts.")
    last_search_query: str | None = Field(default=None, description="Last SearchAction query, if any.")
    last_search_hits: list[str] = Field(
        default_factory=list,
        description="``message_id``s from the last search (scaffold).",
    )
    flags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-message string flags (e.g. {'msg-1': ['starred']}).",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of per-step rewards so far (shaping + graded).",
    )
    sent_messages: list[EmailMessage] = Field(
        default_factory=list,
        description="Outbound mail created via reply/compose in this episode.",
    )


# ---------------------------------------------------------------------------
# Actions (discriminated union on ``action_type``)
# ---------------------------------------------------------------------------


class ReadEmailAction(BaseModel):
    """Open a message and optionally mark it read."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["read_email"] = "read_email"
    message_id: str = Field(min_length=1, description="Which message to open.")
    mark_read: bool = Field(default=True, description="If true, transition unread -> read.")


class ReplyAction(BaseModel):
    """Reply to an existing message."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["reply"] = "reply"
    message_id: str = Field(min_length=1, description="Message being replied to.")
    body: str = Field(description="Reply body (plain text).")
    reply_all: bool = Field(default=False, description="Include all original recipients (scaffold hint).")
    cc_addresses: list[str] = Field(default_factory=list, description="Extra CC addresses for this reply.")


class ComposeAction(BaseModel):
    """Start or continue a new outbound message."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["compose"] = "compose"
    draft_id: str | None = Field(
        default=None,
        description="If set, update this draft; if None, environment may create a new draft id.",
    )
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    bcc_addresses: list[str] = Field(default_factory=list)
    subject: str = Field(default="", description="Outbound subject.")
    body: str = Field(default="", description="Outbound body.")


class LabelAction(BaseModel):
    """Apply or replace folder/labels on a message."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["label"] = "label"
    message_id: str = Field(min_length=1)
    labels: list[str] = Field(
        default_factory=list,
        description="Folder or label names to apply.",
    )
    replace_existing: bool = Field(
        default=True,
        description="If true, replace prior folder assignment; if false, merge (environment-defined).",
    )


class DeleteAction(BaseModel):
    """Remove a message from the active inbox view."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["delete"] = "delete"
    message_id: str = Field(min_length=1)
    permanent: bool = Field(
        default=False,
        description="If false, move to trash; if true, purge (when implemented).",
    )


class SearchAction(BaseModel):
    """Search headers/bodies (simulated)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["search"] = "search"
    query: str = Field(min_length=1, description="Free-text query.")
    folder_scope: StandardFolder | str | None = Field(
        default=None,
        description="Restrict search to a folder; None searches all visible mail.",
    )
    limit: int = Field(default=20, ge=1, le=500, description="Maximum hits to return.")


class NoopAction(BaseModel):
    """Explicit no-operation step (e.g. wait or internal reasoning tick)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["noop"] = "noop"
    reason: str | None = Field(default=None, description="Optional debug note (not graded).")


Action = Annotated[
    Union[
        ReadEmailAction,
        ReplyAction,
        ComposeAction,
        LabelAction,
        DeleteAction,
        SearchAction,
        NoopAction,
    ],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Observation & step transition
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    Agent-visible view after ``reset`` or ``step``.

    ``info`` carries environment diagnostics; rubric data must not appear here.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="Current task id.")
    task_type: str = Field(description="Current task type.")
    prompt: str = Field(description="Instruction text for the agent.")
    inbox_snapshot: list[EmailMessage] = Field(description="Messages visible in the inbox view.")
    focus_message_id: str | None = Field(default=None)
    done: bool = Field(default=False)
    reward: float | None = Field(
        default=None,
        description="Shaping or terminal reward for this transition, if any.",
    )
    step_count: int = Field(default=0, ge=0, description="Environment step counter after this transition.")
    schema_version: str = Field(default="1", description="Observation JSON schema revision.")
    available_action_types: list[str] = Field(
        default_factory=lambda: list(ACTION_TYPE_LITERALS),
        description="Discriminator strings the agent may send on the next step.",
    )
    standard_folders: list[str] = Field(
        default_factory=lambda: list(STANDARD_FOLDERS),
        description="Folder names valid for triage-style tasks.",
    )
    info: dict[str, Any] = Field(default_factory=dict, description="Diagnostics and non-rubric metadata.")


class ResetResponse(BaseModel):
    """Payload returned by ``EmailEnv.reset()`` (and typically ``POST /reset``)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Client session this episode belongs to.")
    episode_id: str = Field(description="Unique id for this episode instance.")
    observation: Observation = Field(description="Initial agent-visible observation.")


class StepResult(BaseModel):
    """Single transition outcome from ``step(action)``."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation = Field(description="Post-step observation.")
    reward: float = Field(
        ge=-1.0,
        le=1.0,
        description="Scalar reward for this step (may be negative e.g. bad delete).",
    )
    done: bool = Field(description="Whether the episode ended on this step.")
    truncated: bool = Field(
        default=False,
        description="True if observation was truncated for token limits (future use).",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. grader name, warnings, search hit counts.",
    )


__all__ = [
    "ACTION_TYPE_LITERALS",
    "STANDARD_FOLDERS",
    "Action",
    "Email",
    "ResetResponse",
    "ComposeAction",
    "DeleteAction",
    "DraftMessage",
    "EmailMessage",
    "InboxState",
    "LabelAction",
    "MessageImportance",
    "NoopAction",
    "Observation",
    "ReadEmailAction",
    "ReplyAction",
    "SearchAction",
    "StandardFolder",
    "StepResult",
    "TaskId",
    "TaskSpec",
    "TaskType",
]
