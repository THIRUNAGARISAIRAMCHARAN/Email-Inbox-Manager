"""
Synthetic inbox generation for benchmark episodes.

Seeded ``random.Random`` only — no third-party data libraries.
Templates are hard-coded pools assembled into ``Email`` (``EmailMessage``) rows.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone

from email_env.models import Email, MessageImportance

# --- Valid episode types for ``generate_inbox`` ---------------------------------
TASK_IDS: frozenset[str] = frozenset({"easy_reply", "medium_triage", "hard_thread"})

# --- Template pools (subjects / bodies / senders) --------------------------------
_NEWSLETTER_SUBJECTS: tuple[str, ...] = (
    "Weekly Digest: Tech & You",
    "Your Sunday Reading List",
    "Deals of the Week — Members Only",
    "Community newsletter — March edition",
    "Product roundup: what shipped this month",
)

_NEWSLETTER_BODIES: tuple[str, ...] = (
    "Hi there,\n\nHere are this week's top stories and links we think you'll enjoy.\n\n— The editorial team",
    "Hello,\n\nQuick digest of community highlights and upcoming events.\n\nUnsubscribe any time.\n",
    "Greetings,\n\nPhotos, comments, and featured posts from people you follow.\n\nThanks for reading.",
)

_FYI_SUBJECTS: tuple[str, ...] = (
    "Parking policy reminder",
    "Office closed Monday",
    "VPN maintenance tonight 11pm",
    "New badge policy for visitors",
    "Coffee machine replaced on 3F",
)

_FYI_BODIES: tuple[str, ...] = (
    "This is an automated notice. No reply needed.\n\n— Workplace Ops",
    "Heads-up for the team. Details are in the wiki.\n\nThanks,\nOps",
    "Short reminder per facilities. Let us know if issues persist.\n",
)

_URGENT_SUBJECTS: tuple[str, ...] = (
    "URGENT: client call moved to 2pm",
    "Action needed: approve expense report",
    "Time-sensitive: sign-off before EOD",
)

_URGENT_BODIES: tuple[str, ...] = (
    "I need a quick confirmation from you so we don't miss the window.\n\nPing me on chat if easier.\n",
    "Can you take a look in the next hour? Legal is waiting on this.\n\nThanks,\nPat",
    "Please confirm you saw this — deadline is tight.\n",
)

_SPAM_SUBJECTS: tuple[str, ...] = (
    "You have WON a prize — click here!!!",
    "URGENT inheritance funds waiting",
    "Pharma discount 90% OFF today only",
    "Your account will be suspended CLICK NOW",
)

_SPAM_BODIES: tuple[str, ...] = (
    "Congratulations!!! Claim your reward at http://totally-legit.example.invalid\n",
    "Dear friend, I need your help transferring funds. Reply with bank details.\n",
    "Limited offer! Act now!!! Unsubscribe buried in footer.\n",
)

_FILLER_NAMES: tuple[str, ...] = (
    "jordan.lee",
    "sam.patel",
    "jamie.ortiz",
    "taylor.kim",
    "casey.nguyen",
    "riley.chen",
    "avery.garcia",
    "quinn.murphy",
)

_DOMAINS: tuple[str, ...] = (
    "contoso.test",
    "fabrikam.test",
    "northwind.test",
    "example.work",
)


def inbox_seed(seed_key: str | None, fallback: int = 42) -> int:
    """Map optional HTTP ``seed_key`` to a non-negative int for ``generate_inbox``."""
    if seed_key is None:
        return fallback
    return int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def generate_inbox(task_id: str, seed: int = 42) -> list[Email]:
    """
    Build a reproducible inbox for the given benchmark task.

    - ``easy_reply``: 8–12 emails; one urgent unread from ``boss@company.com``,
      subject **Re: Project deadline**, asking for a status update.
    - ``medium_triage``: 15–20 emails (newsletters, urgent, spam, FYI); exactly
      three marked as needing a reply; **all unread**.
    - ``hard_thread``: 20–25 emails including a 4-message thread about
      **Q4 budget meeting** (chained ``in_reply_to`` / shared ``thread_id``).
    """
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id {task_id!r}; expected one of {sorted(TASK_IDS)}")

    rng = random.Random(seed)
    if task_id == "easy_reply":
        return _generate_easy_reply(rng)
    if task_id == "medium_triage":
        return _generate_medium_triage(rng)
    return _generate_hard_thread(rng)


def _newest_first(emails: list[Email]) -> list[Email]:
    return sorted(emails, key=lambda e: e.date_sent, reverse=True)


def _addr(rng: random.Random, local: str | None = None) -> str:
    if local is None:
        local = rng.choice(_FILLER_NAMES)
    return f"{local}@{rng.choice(_DOMAINS)}"


def _mid(rng: random.Random, tag: str) -> str:
    return f"{tag}-{rng.getrandbits(40):010x}"


def _time_series(rng: random.Random, start: datetime, n: int) -> list[datetime]:
    out: list[datetime] = []
    t = start
    for _ in range(n):
        t = t + timedelta(minutes=rng.randint(1, 45))
        out.append(t)
    return out


def _generate_easy_reply(rng: random.Random) -> list[Email]:
    n = rng.randint(8, 12)
    base = datetime(2025, 3, 3, 9, 0, tzinfo=timezone.utc)
    times = _time_series(rng, base, n)
    boss_time = times[-1] + timedelta(minutes=rng.randint(5, 30))

    boss_id = _mid(rng, "boss")
    boss = Email(
        message_id=boss_id,
        subject="Re: Project deadline",
        body=(
            "Hi,\n\n"
            "Following up from this morning — can you send a brief status update on the deliverable "
            "before the leadership sync? They want blockers, ETA, and anything you need from my side.\n\n"
            "Thanks,\nAlexDirector\nVP Engineering\n"
        ),
        from_address="boss@company.com",
        to_addresses=["me@company.com"],
        date_sent=boss_time,
        read=False,
        importance=MessageImportance.high,
        thread_id=f"deadline-thread-{boss_id[-6:]}",
        folder="inbox",
        metadata={
            "synthetic": True,
            "topic": "project deadline",
            "priority": "high",
            "role": "boss_request",
        },
    )

    fillers: list[Email] = []
    pool_subjects = list(_FYI_SUBJECTS) + list(_NEWSLETTER_SUBJECTS)
    pool_bodies = list(_FYI_BODIES) + list(_NEWSLETTER_BODIES)
    for i in range(n - 1):
        t = times[i]
        sid = _mid(rng, "fill")
        fillers.append(
            Email(
                message_id=sid,
                subject=f"{rng.choice(pool_subjects)} (#{i + 1})",
                body=rng.choice(pool_bodies),
                from_address=_addr(rng),
                to_addresses=["me@company.com"],
                date_sent=t,
                read=rng.random() < 0.35,
                importance=MessageImportance.normal,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": rng.choice(["office", "newsletter", "operations", "facilities"]),
                    "priority": rng.choice(["low", "medium"]),
                    "folder": rng.choice(["inbox", "updates"]),
                },
            )
        )

    return _newest_first([boss] + fillers)


def _generate_medium_triage(rng: random.Random) -> list[Email]:
    n = rng.randint(15, 20)
    base = datetime(2025, 3, 5, 7, 30, tzinfo=timezone.utc)
    times = _time_series(rng, base, n + 8)

    triage_folder = rng.choice(["finance", "updates", "social"])
    focus_id = _mid(rng, "triage-focus")
    core: list[Email] = []

    core.append(
        Email(
            message_id=focus_id,
            subject="Invoice #4481 — payment receipt attached",
            body=(
                "Accounting team — please file this under the Q1 vendor bucket.\n"
                "Let me know if anything is missing.\n\nRegards,\nMorgan\nAP Specialist\n"
            ),
            from_address="ap.notices@fabrikam.test",
            to_addresses=["me@company.com"],
            cc_addresses=["finance.team@fabrikam.test"],
            date_sent=times[0],
            read=False,
            importance=MessageImportance.normal,
            folder="inbox",
            metadata={
                "synthetic": True,
                "topic": "invoice",
                "priority": "medium",
                "folder": triage_folder,
                "needs_reply": False,
            },
        )
    )

    reply_prompts = (
        ("Quick question on your deck", "Do you have five minutes to sanity-check slides 4–6 before I send?"),
        ("Budget line item", "Can you confirm we should still code this under 4020 for March?"),
        ("Design review follow-up", "Should we land the new layout in sprint 12 or push to 13?"),
    )
    for i, (subj, para) in enumerate(reply_prompts, start=1):
        core.append(
            Email(
                message_id=_mid(rng, "needs-reply"),
                subject=subj,
                body=f"Hi,\n\n{para}\n\nThanks!\n{rng.choice(_FILLER_NAMES).replace('.', ' ').title()}",
                from_address=_addr(rng),
                to_addresses=["me@company.com"],
                date_sent=times[i],
                read=False,
                importance=MessageImportance.normal,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": "coordination",
                    "priority": "medium",
                    "folder": "inbox",
                    "needs_reply": True,
                },
            )
        )

    pool: list[Email] = []

    for _ in range(rng.randint(4, 6)):
        pool.append(
            Email(
                message_id=_mid(rng, "news"),
                subject=rng.choice(_NEWSLETTER_SUBJECTS),
                body=rng.choice(_NEWSLETTER_BODIES),
                from_address=f"digest@{rng.choice(['mail', 'info', 'news'])}.contoso.test",
                to_addresses=["me@company.com"],
                date_sent=times[len(pool) + len(core)],
                read=False,
                importance=MessageImportance.low,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": "newsletter",
                    "priority": "low",
                    "folder": "updates",
                    "needs_reply": False,
                    "category": "newsletter",
                },
            )
        )

    for _ in range(rng.randint(2, 4)):
        pool.append(
            Email(
                message_id=_mid(rng, "urgent"),
                subject=rng.choice(_URGENT_SUBJECTS),
                body=rng.choice(_URGENT_BODIES),
                from_address=_addr(rng),
                to_addresses=["me@company.com"],
                date_sent=times[len(pool) + len(core)],
                read=False,
                importance=MessageImportance.high,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": "urgent",
                    "priority": "high",
                    "folder": "inbox",
                    "needs_reply": False,
                },
            )
        )

    for _ in range(rng.randint(3, 5)):
        pool.append(
            Email(
                message_id=_mid(rng, "spam"),
                subject=rng.choice(_SPAM_SUBJECTS),
                body=rng.choice(_SPAM_BODIES),
                from_address=f"promo{rng.randint(1, 999)}@suspicious-bulk.example.invalid",
                to_addresses=["me@company.com"],
                date_sent=times[len(pool) + len(core)],
                read=False,
                importance=MessageImportance.low,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": "spam",
                    "priority": "low",
                    "folder": "social",
                    "needs_reply": False,
                    "category": "spam",
                },
            )
        )

    while len(core) + len(pool) < n:
        pool.append(
            Email(
                message_id=_mid(rng, "fyi"),
                subject=rng.choice(_FYI_SUBJECTS),
                body=rng.choice(_FYI_BODIES),
                from_address="noreply@northwind.test",
                to_addresses=["me@company.com"],
                date_sent=times[len(pool) + len(core)],
                read=False,
                importance=MessageImportance.low,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": "notice",
                    "priority": "low",
                    "folder": "inbox",
                    "needs_reply": False,
                },
            )
        )

    combined = core + pool
    if len(combined) > n:
        protected = set(m.message_id for m in core)
        rest = [e for e in combined if e.message_id not in protected]
        rng.shuffle(rest)
        combined = core + rest[: n - len(core)]

    emails = _newest_first(combined)
    latest = emails[0].date_sent
    focus_msg = next(e for e in emails if e.message_id == focus_id)
    idx = emails.index(focus_msg)
    bumped = focus_msg.model_copy(update={"date_sent": latest + timedelta(seconds=rng.randint(30, 180))})
    emails[idx] = bumped
    emails = _newest_first(emails)

    assert len(emails) == n
    assert sum(1 for e in emails if e.metadata.get("needs_reply")) == 3
    assert all(e.read is False for e in emails)
    return emails


def _generate_hard_thread(rng: random.Random) -> list[Email]:
    n = rng.randint(20, 25)
    base = datetime(2025, 2, 20, 14, 0, tzinfo=timezone.utc)
    thread_id = _mid(rng, "q4-thread")

    t0 = base
    id1 = _mid(rng, "q4-a")
    id2 = _mid(rng, "q4-b")
    id3 = _mid(rng, "q4-c")
    id4 = _mid(rng, "q4-d")

    thread_topic = "q4 budget meeting"

    m1 = Email(
        message_id=id1,
        subject="Q4 budget meeting",
        body=(
            "Team,\n\n"
            "Scheduling the cross-functional Q4 budget meeting for next Thursday at 10am PT. "
            "Please review the pre-read spreadsheet and come with 1–2 risks.\n\n"
            "— Dana (Finance)\n"
        ),
        from_address="dana.price@contoso.test",
        to_addresses=["me@company.com", "engineering.leads@contoso.test"],
        date_sent=t0,
        read=False,
        importance=MessageImportance.normal,
        thread_id=thread_id,
        in_reply_to=None,
        folder="inbox",
        metadata={"synthetic": True, "topic": thread_topic, "thread_pos": 1, "priority": "medium", "folder": "inbox"},
    )
    m2 = Email(
        message_id=id2,
        subject="Re: Q4 budget meeting",
        body=(
            "Thanks Dana — Engineering will bring headcount forecast deltas and infra carry-over.\n\n"
            "One clarification: should we include contractor burn in the same slide deck?\n\n"
            "Chris\n"
        ),
        from_address="chris.rivera@contoso.test",
        to_addresses=["me@company.com", "dana.price@contoso.test"],
        cc_addresses=["engineering.leads@contoso.test"],
        date_sent=t0 + timedelta(hours=2),
        read=False,
        importance=MessageImportance.normal,
        thread_id=thread_id,
        in_reply_to=id1,
        folder="inbox",
        metadata={"synthetic": True, "topic": thread_topic, "thread_pos": 2, "priority": "medium", "folder": "inbox"},
    )
    m3 = Email(
        message_id=id3,
        subject="Re: Q4 budget meeting",
        body=(
            "Yes, include contractors on the same deck but as a separate subsection — easier for execs to scan.\n\n"
            "I'll add a tab for assumptions.\n\n"
            "Dana\n"
        ),
        from_address="dana.price@contoso.test",
        to_addresses=["me@company.com", "chris.rivera@contoso.test"],
        date_sent=t0 + timedelta(hours=5),
        read=False,
        importance=MessageImportance.normal,
        thread_id=thread_id,
        in_reply_to=id2,
        folder="inbox",
        metadata={"synthetic": True, "topic": thread_topic, "thread_pos": 3, "priority": "medium", "folder": "inbox"},
    )
    m4 = Email(
        message_id=id4,
        subject="Re: Q4 budget meeting",
        body=(
            "Looping you in — before we lock slide 4, can you confirm whether we're assuming the delayed "
            "vendor payment hits in Q4 or Q1? That changes the narrative for the risk callout.\n\n"
            "What we need from you: one sentence on timing + any blocker for the vote.\n\n"
            "Thanks,\nChris\n"
        ),
        from_address="chris.rivera@contoso.test",
        to_addresses=["me@company.com"],
        cc_addresses=["dana.price@contoso.test"],
        date_sent=t0 + timedelta(hours=8),
        read=False,
        importance=MessageImportance.high,
        thread_id=thread_id,
        in_reply_to=id3,
        folder="inbox",
        metadata={
            "synthetic": True,
            "topic": thread_topic,
            "thread_pos": 4,
            "priority": "high",
            "folder": "inbox",
            "thread_continuation": True,
        },
    )

    thread_msgs = [m1, m2, m3, m4]
    emails: list[Email] = list(thread_msgs)

    filler_topics = (
        "invoice",
        "meeting",
        "support",
        "newsletter",
        "security alert",
        "operations",
        "facilities",
        "recruiting",
        "design critique",
        "customer escalation",
    )

    times = _time_series(rng, t0 - timedelta(days=3), max(0, n - len(emails)))
    idx = 0
    while len(emails) < n:
        topic = rng.choice(filler_topics)
        emails.append(
            Email(
                message_id=_mid(rng, "fill"),
                subject=f"[{topic}] unrelated note #{len(emails)}",
                body=(
                    f"This message is outside the Q4 thread — regarding {topic}. "
                    "No action required unless you're the owner.\n"
                ),
                from_address=_addr(rng),
                to_addresses=["me@company.com"],
                date_sent=times[idx % len(times)] if times else base,
                read=False,
                importance=rng.choice([MessageImportance.low, MessageImportance.normal]),
                thread_id=None,
                folder="inbox",
                metadata={
                    "synthetic": True,
                    "topic": topic,
                    "priority": rng.choice(["low", "medium"]),
                    "folder": rng.choice(["inbox", "updates", "finance", "social"]),
                },
            )
        )
        idx += 1

    return _newest_first(emails)


__all__ = ["TASK_IDS", "generate_inbox", "inbox_seed"]
