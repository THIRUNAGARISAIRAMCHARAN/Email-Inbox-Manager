"""
State-based graders: ``get_grader(task_id)`` returns a deterministic score in [0.0, 1.0].

Each grader returns **partial credit** (continuous or multi-factor), not only 0/1.
No randomness: same ``GraderContext`` always yields the same float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from email_env.models import EmailMessage

GraderCallable = Callable[["GraderContext"], float]


def _clamp01(x: float) -> float:
    """Inclusive bounds [0.0, 1.0], stable rounding for JSON/logging."""
    return round(min(1.0, max(0.0, float(x))), 6)


@dataclass(frozen=True)
class GraderContext:
    """Immutable snapshot passed to task graders (built by ``EmailEnv``)."""

    task_id: str
    emails: tuple[EmailMessage, ...]
    sent_messages: tuple[EmailMessage, ...]
    initial_emails: tuple[EmailMessage, ...]
    trash_message_ids: frozenset[str]


def _by_id(emails: tuple[EmailMessage, ...], message_id: str) -> EmailMessage | None:
    for e in emails:
        if e.message_id == message_id:
            return e
    return None


def _boss_message_id(initial: tuple[EmailMessage, ...]) -> str | None:
    for e in initial:
        if e.from_address.strip().lower() == "boss@company.com":
            return e.message_id
    return None


def easy_reply_grader(ctx: GraderContext) -> float:
    """
    Partial credit (continuous):
    - Reading the boss message contributes up to ~0.22.
    - Replying to the boss contributes up to ~0.78, scaled smoothly by body length
      (starts crediting above ~20 chars, approaches 1.0 combined when read + long reply).

    Always in [0.0, 1.0]; deterministic.
    """
    if ctx.task_id != "easy_reply":
        return _clamp01(0.0)
    bid = _boss_message_id(ctx.initial_emails)
    if bid is None:
        return _clamp01(0.0)

    boss = _by_id(ctx.emails, bid)
    read_s = 1.0 if boss is not None and boss.read else 0.0

    best_len = 0
    for s in ctx.sent_messages:
        if s.metadata.get("kind") != "reply":
            continue
        parent = s.in_reply_to or s.metadata.get("reply_to_message_id")
        if parent != bid:
            continue
        best_len = max(best_len, len(s.body.strip()))

    # Length curve: no credit at <=20 chars, full component at 100+ chars
    if best_len <= 20:
        reply_q = 0.0
    else:
        reply_q = min(1.0, (best_len - 20.0) / 80.0) ** 0.88

    score = 0.22 * read_s + 0.78 * reply_q
    return _clamp01(score)


def _newsletter_labelled(e: EmailMessage) -> bool:
    if str(e.folder).lower() == "updates":
        return True
    labels = e.metadata.get("labels")
    if isinstance(labels, list):
        for lab in labels:
            if str(lab).lower() in ("updates", "newsletter"):
                return True
    return False


def medium_triage_grader(ctx: GraderContext) -> float:
    """
    Partial credit: 0.33×newsletters + 0.33×needs-reply replies + 0.34×spam removed.
    Each factor is a fraction in [0, 1] (or 1.0 if class empty). Deterministic.
    """
    if ctx.task_id != "medium_triage":
        return _clamp01(0.0)

    newsletter_ids = [e.message_id for e in ctx.initial_emails if e.metadata.get("category") == "newsletter"]
    needs_ids = [e.message_id for e in ctx.initial_emails if e.metadata.get("needs_reply") is True]
    spam_ids = [e.message_id for e in ctx.initial_emails if e.metadata.get("category") == "spam"]

    inbox_ids = {e.message_id for e in ctx.emails}

    if newsletter_ids:
        num_ok = 0
        for nid in sorted(newsletter_ids):
            cur = _by_id(ctx.emails, nid)
            if cur is not None and _newsletter_labelled(cur):
                num_ok += 1
        a = num_ok / len(newsletter_ids)
    else:
        a = 1.0

    if needs_ids:
        replied = 0
        for nid in sorted(needs_ids):
            for s in ctx.sent_messages:
                if s.metadata.get("kind") != "reply":
                    continue
                parent = s.in_reply_to or s.metadata.get("reply_to_message_id")
                if parent == nid:
                    replied += 1
                    break
        b = replied / len(needs_ids)
    else:
        b = 1.0

    if spam_ids:
        removed = 0
        for sid in sorted(spam_ids):
            if sid not in inbox_ids:
                removed += 1
        c = removed / len(spam_ids)
    else:
        c = 1.0

    score = (33 * a + 33 * b + 34 * c) / 100.0
    return _clamp01(score)


_THREAD_TOPIC = "q4 budget meeting"


def _thread_ids(initial: tuple[EmailMessage, ...]) -> frozenset[str]:
    out: set[str] = set()
    for e in initial:
        if str(e.metadata.get("topic", "")).lower() != _THREAD_TOPIC:
            continue
        if e.thread_id:
            out.add(e.message_id)
    return frozenset(out)


def _mentions_q4_budget_meeting(text: str) -> bool:
    t = text.lower()
    return "q4" in t or "budget" in t or "meeting" in t


def _qualifying_thread_send_score(ctx: GraderContext) -> float:
    """Partial 0–1 for outbound quality when not yet fully qualifying."""
    best = 0.0
    for s in ctx.sent_messages:
        if s.metadata.get("kind") not in ("reply", "compose"):
            continue
        blob = f"{s.subject}\n{s.body}"
        L = len(s.body.strip())
        kw = _mentions_q4_budget_meeting(blob)
        len_s = min(1.0, max(0.0, (L - 1.0) / 99.0)) if L > 0 else 0.0
        kw_s = 1.0 if kw else 0.0
        # blend length and keywords for smooth partial before strict threshold
        cand = 0.55 * len_s + 0.45 * kw_s * len_s
        best = max(best, cand)
    return _clamp01(best)


def _qualifying_thread_send(ctx: GraderContext) -> bool:
    for s in ctx.sent_messages:
        if s.metadata.get("kind") not in ("reply", "compose"):
            continue
        blob = f"{s.subject}\n{s.body}"
        if len(s.body.strip()) > 50 and _mentions_q4_budget_meeting(blob):
            return True
    return False


def hard_thread_grader(ctx: GraderContext) -> float:
    """
    Partial credit:
    - Full 1.0 when a send has body > 50 and mentions Q4/budget/meeting (same threshold as before).
    - Otherwise smooth score from (fraction of thread messages read) plus a small boost from
      ``near-miss`` sends (length + keyword overlap), capped below 1.0 until the strict bar is met.

    Deterministic; values in [0.0, 1.0].
    """
    if ctx.task_id != "hard_thread":
        return _clamp01(0.0)

    tids = _thread_ids(ctx.initial_emails)
    if not tids:
        return _clamp01(0.0)

    if _qualifying_thread_send(ctx):
        return _clamp01(1.0)

    n = len(tids)
    read_n = 0
    for mid in sorted(tids):
        cur = _by_id(ctx.emails, mid)
        if cur is not None and cur.read:
            read_n += 1
    f = read_n / n if n else 0.0

    # Smooth read-through partial (dominant when browsing thread)
    read_component = (f**1.05) * 0.72
    send_partial = _qualifying_thread_send_score(ctx) * 0.26
    score = read_component + send_partial + 0.02 * f
    # Cap below 1.0 until strict qualifying send (length + keywords) is met
    return _clamp01(min(0.985, score))


_GRADERS: dict[str, GraderCallable] = {
    "easy_reply": easy_reply_grader,
    "medium_triage": medium_triage_grader,
    "hard_thread": hard_thread_grader,
}


def get_grader(task_id: str) -> GraderCallable:
    """Return the state-based scorer for ``task_id`` (raises ``KeyError`` if unknown)."""
    if task_id not in _GRADERS:
        raise KeyError(f"Unknown task_id for grader: {task_id!r}; known: {sorted(_GRADERS)}")
    return _GRADERS[task_id]


def list_grader_task_ids() -> list[str]:
    return sorted(_GRADERS.keys())


__all__ = [
    "GraderCallable",
    "GraderContext",
    "easy_reply_grader",
    "get_grader",
    "hard_thread_grader",
    "list_grader_task_ids",
    "medium_triage_grader",
]
