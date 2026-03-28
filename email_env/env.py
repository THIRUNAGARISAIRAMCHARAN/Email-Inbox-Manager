"""
OpenEnv email inbox: ``EmailEnv`` with reset / step / state and action handlers.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone

from email_env.data_gen import generate_inbox
from email_env.graders import GraderContext, get_grader
from email_env.models import (
    Action,
    ComposeAction,
    DeleteAction,
    EmailMessage,
    InboxState,
    LabelAction,
    MessageImportance,
    NoopAction,
    Observation,
    ReadEmailAction,
    ReplyAction,
    ResetResponse,
    SearchAction,
    StepResult,
    TaskSpec,
)
from email_env.tasks import get_task_builder, list_task_types

_DEFAULT_MAX_STEPS = 20


class EmailEnv:
    """
    Stateful simulated inbox for one session.

    Thread-safety: use one instance per ``session_id`` in the HTTP layer, guarded
    by a per-session lock, or call methods from a single thread.
    """

    def __init__(self, task_id: str, session_id: str, seed: int = 42) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.seed = seed
        self._builder = get_task_builder(task_id)
        self._lock = threading.RLock()

        self._episode_id: str = ""
        self._emails: list[EmailMessage] = []
        self._initial_emails: tuple[EmailMessage, ...] = ()
        self._sent_messages: list[EmailMessage] = []
        self._focus_message_id: str | None = None
        self._current_task: TaskSpec | None = None
        self._max_steps_effective: int = _DEFAULT_MAX_STEPS
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_graded_score: float | None = None
        self._holistic_score_prev: float = 0.0

        self._search_filter_ids: frozenset[str] | None = None
        self._last_search_query: str | None = None
        self._last_search_hits: list[str] = []
        self._trash_message_ids: list[str] = []
        self._episode_serial: int = 0
        self._msg_serial: int = 0

    def _next_episode_id(self) -> str:
        self._episode_serial += 1
        raw = f"{self.session_id}|{self.seed}|ep|{self._episode_serial}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _next_sent_id(self, prefix: str) -> str:
        self._msg_serial += 1
        raw = f"{self.session_id}|{self._episode_id}|{prefix}|{self._msg_serial}".encode()
        return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:14]}"

    def reset(self) -> ResetResponse:
        with self._lock:
            self._episode_id = self._next_episode_id()
            self._msg_serial = 0
            self._emails = list(generate_inbox(self.task_id, seed=self.seed))
            self._initial_emails = tuple(m.model_copy(deep=True) for m in self._emails)
            self._sent_messages = []
            self._cumulative_reward = 0.0
            self._step_count = 0
            self._done = False
            self._last_graded_score = None
            self._holistic_score_prev = 0.0
            self._search_filter_ids = None
            self._last_search_query = None
            self._last_search_hits = []
            self._trash_message_ids = []
            self._focus_message_id = self._emails[0].message_id if self._emails else None
            self._current_task = self._builder(self._emails, self._episode_id)
            ms = self._current_task.max_steps
            self._max_steps_effective = int(ms) if ms is not None else _DEFAULT_MAX_STEPS
            obs = self._build_observation(reward=None)
            return ResetResponse(
                session_id=self.session_id,
                episode_id=self._episode_id,
                observation=obs,
            )

    def step(self, action: Action) -> StepResult:
        with self._lock:
            if self._current_task is None:
                raise RuntimeError("Call reset() before step()")

            if self._done:
                obs = self._build_observation(reward=self._last_graded_score)
                return StepResult(
                    observation=obs,
                    reward=0.0,
                    done=True,
                    info={"warning": "episode_already_done", "cumulative_reward": self._cumulative_reward},
                )

            self._step_count += 1
            info: dict = {}

            if isinstance(action, NoopAction):
                pass

            elif isinstance(action, ReadEmailAction):
                info.update(self._handle_read(action))

            elif isinstance(action, ReplyAction):
                info.update(self._handle_reply(action))

            elif isinstance(action, ComposeAction):
                info.update(self._handle_compose(action))

            elif isinstance(action, LabelAction):
                info.update(self._handle_label(action))

            elif isinstance(action, DeleteAction):
                info.update(self._handle_delete(action))

            elif isinstance(action, SearchAction):
                info.update(self._handle_search(action))

            else:
                info["warning"] = "unhandled_action"

            ctx = self._make_grader_context()
            holistic = float(get_grader(self.task_id)(ctx))
            step_reward = round(holistic - self._holistic_score_prev, 6)
            self._holistic_score_prev = holistic
            info["holistic_score"] = holistic
            info["grader"] = self.task_id

            self._cumulative_reward += step_reward
            info["cumulative_reward"] = self._cumulative_reward

            term: str | None = None
            if holistic >= 1.0 - 1e-9:
                self._last_graded_score = holistic
                self._done = True
                term = "grader_success"
            if self._step_count >= self._max_steps_effective:
                self._done = True
                term = "max_steps" if term is None else f"{term}+max_steps"
            if term:
                info["terminated"] = term

            obs = self._build_observation(reward=step_reward)
            if not isinstance(action, SearchAction):
                self._search_filter_ids = None

            return StepResult(
                observation=obs,
                reward=max(-1.0, min(1.0, step_reward)),
                done=self._done,
                info=info,
            )

    def _make_grader_context(self) -> GraderContext:
        return GraderContext(
            task_id=self.task_id,
            emails=tuple(self._emails),
            sent_messages=tuple(self._sent_messages),
            initial_emails=self._initial_emails,
            trash_message_ids=frozenset(self._trash_message_ids),
        )

    def state(self) -> InboxState:
        with self._lock:
            return InboxState(
                session_episode_id=self._episode_id,
                task_type=self.task_id,
                emails=list(self._emails),
                focus_message_id=self._focus_message_id,
                current_task=self._current_task,
                step_count=self._step_count,
                done=self._done,
                last_reward=self._last_graded_score,
                trash_message_ids=list(self._trash_message_ids),
                folder_index=self._folder_index(),
                last_search_query=self._last_search_query,
                last_search_hits=list(self._last_search_hits),
                cumulative_reward=self._cumulative_reward,
                sent_messages=list(self._sent_messages),
            )

    def _index_of(self, message_id: str) -> int | None:
        for i, e in enumerate(self._emails):
            if e.message_id == message_id:
                return i
        return None

    def _handle_read(self, action: ReadEmailAction) -> dict:
        info: dict = {}
        idx = self._index_of(action.message_id)
        if idx is None:
            info["read_error"] = "message_not_found"
            return info
        m = self._emails[idx]
        if action.mark_read and not m.read:
            self._emails[idx] = m.model_copy(update={"read": True})
        return info

    def _handle_reply(self, action: ReplyAction) -> dict:
        info: dict = {}
        idx = self._index_of(action.message_id)
        if idx is None:
            info["reply_error"] = "parent_not_found"
            return info
        parent = self._emails[idx]
        sent = self._make_sent_reply(action, parent)
        self._sent_messages.append(sent)
        info["sent_message_id"] = sent.message_id
        return info

    def _handle_compose(self, action: ComposeAction) -> dict:
        info: dict = {}
        sent = self._make_sent_compose(action)
        self._sent_messages.append(sent)
        info["sent_message_id"] = sent.message_id
        return info

    def _handle_label(self, action: LabelAction) -> dict:
        info: dict = {}
        idx = self._index_of(action.message_id)
        if idx is None:
            info["label_error"] = "message_not_found"
            return info
        m = self._emails[idx]
        primary = action.labels[0] if action.labels else m.folder
        new_meta = dict(m.metadata)
        new_meta["labels"] = list(action.labels)
        self._emails[idx] = m.model_copy(update={"folder": str(primary), "metadata": new_meta})
        return info

    def _handle_delete(self, action: DeleteAction) -> dict:
        info: dict = {}
        idx = self._index_of(action.message_id)
        if idx is None:
            info["delete_error"] = "message_not_found"
            return info

        m = self._emails[idx]
        ctx_focus = self._current_task.context.get("focus_message_id") if self._current_task else None
        if ctx_focus and m.message_id == ctx_focus:
            info["delete_blocked"] = "focus_protected"
            return info

        removed_id = m.message_id
        del self._emails[idx]
        if not action.permanent:
            self._trash_message_ids.append(removed_id)
        return info

    def _handle_search(self, action: SearchAction) -> dict:
        info: dict = {}
        q = action.query.strip().lower()
        pool = self._emails
        if action.folder_scope:
            pool = [e for e in pool if str(e.folder) == str(action.folder_scope)]
        hits: list[str] = []
        for e in pool:
            blob = f"{e.subject} {e.body} {e.from_address}".lower()
            if q in blob:
                hits.append(e.message_id)
                if len(hits) >= action.limit:
                    break
        self._last_search_query = action.query
        self._last_search_hits = hits
        self._search_filter_ids = frozenset(hits)
        info["search_hit_count"] = len(hits)
        info["search_hit_message_ids"] = hits
        return info

    def _make_sent_reply(self, action: ReplyAction, parent: EmailMessage) -> EmailMessage:
        mid = self._next_sent_id("sent-reply")
        subj = parent.subject if parent.subject.lower().startswith("re:") else f"Re: {parent.subject}"
        return EmailMessage(
            message_id=mid,
            subject=subj[:500],
            body=action.body,
            from_address="me@company.com",
            to_addresses=[parent.from_address],
            cc_addresses=list(action.cc_addresses),
            date_sent=datetime.now(timezone.utc),
            read=True,
            thread_id=parent.thread_id,
            in_reply_to=parent.message_id,
            importance=MessageImportance.normal,
            metadata={
                "sent": True,
                "kind": "reply",
                "reply_to_message_id": parent.message_id,
            },
        )

    def _make_sent_compose(self, action: ComposeAction) -> EmailMessage:
        mid = self._next_sent_id("sent-compose")
        subj = action.subject.strip() or "(no subject)"
        return EmailMessage(
            message_id=mid,
            subject=subj,
            body=action.body,
            from_address="me@company.com",
            to_addresses=list(action.to_addresses),
            cc_addresses=list(action.cc_addresses),
            bcc_addresses=list(action.bcc_addresses),
            date_sent=datetime.now(timezone.utc),
            read=True,
            metadata={"sent": True, "kind": "compose"},
        )

    def _snapshot_emails(self) -> list[EmailMessage]:
        if self._search_filter_ids is None:
            return list(self._emails)
        return [e for e in self._emails if e.message_id in self._search_filter_ids]

    def _folder_index(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in self._emails:
            f = str(e.folder)
            out.setdefault(f, []).append(e.message_id)
        return out

    def _build_observation(self, *, reward: float | None) -> Observation:
        assert self._current_task is not None
        return Observation(
            task_id=str(self._current_task.task_id),
            task_type=str(self._current_task.task_type),
            prompt=self._current_task.prompt,
            inbox_snapshot=self._snapshot_emails(),
            focus_message_id=self._focus_message_id,
            done=self._done,
            reward=reward,
            step_count=self._step_count,
            info={
                "step_count": self._step_count,
                "max_steps": self._max_steps_effective,
                "cumulative_reward": self._cumulative_reward,
                "session_id": self.session_id,
                "holistic_score": self._holistic_score_prev,
                **({"last_search_hits": list(self._last_search_hits)} if self._last_search_hits else {}),
            },
        )

    @staticmethod
    def available_task_types() -> list[str]:
        return list_task_types()


EmailInboxEnv = EmailEnv
