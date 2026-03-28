"""
Email Inbox Manager — LLM agent loop against the local OpenEnv HTTP API.

Environment server (httpx): reset / step / tasks
LLM (OpenAI-compatible client): router or OpenAI

Env vars:
  API_BASE_URL   (default: https://router.huggingface.co/v1)
  HF_TOKEN or API_KEY
  MODEL_NAME     (required)
  ENV_BASE_URL   (default: http://localhost:7860)
  INFERENCE_WALL_LIMIT_SEC  (default: 1140 — 19 minutes, under the 20-minute target)
  OPENAI_TIMEOUT_SEC        (default: 45)
  INFERENCE_SEED            (default: 42 — passed to POST /reset)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

TASK_ORDER = ["easy_reply", "medium_triage", "hard_thread"]

SYSTEM_PROMPT = """You control a simulated email inbox via JSON actions only.

Output EXACTLY one JSON object and nothing else — no markdown fences, no commentary.

Required key: "action_type" — one of:
  read_email | reply | compose | label | delete | search | noop

Field names must use snake_case. Use "message_id" for mail identifiers (NOT "email_id", unless you have no choice — the system maps email_id -> message_id if present).

Shapes:
  {"action_type":"read_email","message_id":"<id>","mark_read":true}
  {"action_type":"reply","message_id":"<id>","body":"<plain text>","reply_all":false,"cc_addresses":[]}
  {"action_type":"compose","to_addresses":[],"cc_addresses":[],"bcc_addresses":[],"subject":"","body":"","draft_id":null}
  {"action_type":"label","message_id":"<id>","labels":["updates"],"replace_existing":true}
  {"action_type":"delete","message_id":"<id>","permanent":false}
  {"action_type":"search","query":"<substring>","folder_scope":null,"limit":20}
  {"action_type":"noop","reason":null}

Prefer productive actions (read relevant mail, reply, label, search) over noop when the task requires progress.
""".strip()

VALID_ACTIONS = frozenset(
    {"read_email", "reply", "compose", "label", "delete", "search", "noop"}
)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and v.strip() != "":
        return v
    return default


def _compact_email(m: dict[str, Any], body_max: int = 400) -> dict[str, Any]:
    body = str(m.get("body", ""))
    if len(body) > body_max:
        body = body[:body_max] + "…"
    return {
        "message_id": m.get("message_id"),
        "subject": m.get("subject"),
        "from_address": m.get("from_address"),
        "read": m.get("read"),
        "folder": m.get("folder"),
        "snippet": m.get("snippet"),
        "metadata": m.get("metadata"),
        "body": body,
        "thread_id": m.get("thread_id"),
    }


def _compact_observation(obs: dict[str, Any], inbox_limit: int = 25) -> dict[str, Any]:
    inbox = obs.get("inbox_snapshot") or []
    if len(inbox) > inbox_limit:
        inbox = inbox[:inbox_limit]
    return {
        "task_id": obs.get("task_id"),
        "task_type": obs.get("task_type"),
        "prompt": obs.get("prompt"),
        "focus_message_id": obs.get("focus_message_id"),
        "done": obs.get("done"),
        "step_count": obs.get("step_count"),
        "standard_folders": obs.get("standard_folders"),
        "info": obs.get("info"),
        "inbox_snapshot": [_compact_email(m) for m in inbox],
    }


def _build_user_prompt(
    task_id: str,
    obs: dict[str, Any],
    step_idx: int,
    max_steps: int,
    last_step: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    parts.append(f"Episode task_id: {task_id}")
    parts.append(f"Step: {step_idx + 1} / {max_steps} (stop early if observation.done becomes true).")
    parts.append("")
    parts.append("Current observation (JSON):")
    parts.append(json.dumps(_compact_observation(obs), indent=2))
    if last_step is not None:
        parts.append("")
        parts.append("Last step result (JSON):")
        summary = {
            "done": last_step.get("done"),
            "reward": last_step.get("reward"),
            "truncated": last_step.get("truncated"),
            "info": last_step.get("info"),
        }
        parts.append(json.dumps(summary, indent=2))
        parts.append("(Next observation is included above — it reflects state after that step.)")
    parts.append("")
    parts.append("Respond with the next action as a single JSON object only.")
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].lstrip()
            if block.startswith("{"):
                text = block
                break

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Map common LLM mistakes; coerce message_id to str."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = str(k).strip()
        if key in ("emailId", "EmailId"):
            key = "message_id"
        if key == "email_id" or key == "messageId":
            key = "message_id"
        out[key] = v

    if "message_id" in out and out["message_id"] is not None:
        out["message_id"] = str(out["message_id"])

    at = out.get("action_type")
    if at is not None:
        out["action_type"] = str(at).strip().lower().replace("-", "_")

    return out


def _coerce_action(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {"action_type": "noop"}
    d = _normalize_keys(raw)
    t = d.get("action_type", "noop")
    if t not in VALID_ACTIONS:
        return {"action_type": "noop"}

    if t == "read_email":
        mid = d.get("message_id")
        if not mid:
            return {"action_type": "noop"}
        return {
            "action_type": "read_email",
            "message_id": str(mid),
            "mark_read": bool(d.get("mark_read", True)),
        }

    if t == "reply":
        mid = d.get("message_id")
        body = str(d.get("body", "")).strip()
        if not mid or not body:
            return {"action_type": "noop"}
        return {
            "action_type": "reply",
            "message_id": str(mid),
            "body": body,
            "reply_all": bool(d.get("reply_all", False)),
            "cc_addresses": list(d.get("cc_addresses") or []),
        }

    if t == "compose":
        return {
            "action_type": "compose",
            "draft_id": d.get("draft_id"),
            "to_addresses": list(d.get("to_addresses") or []),
            "cc_addresses": list(d.get("cc_addresses") or []),
            "bcc_addresses": list(d.get("bcc_addresses") or []),
            "subject": str(d.get("subject", "")),
            "body": str(d.get("body", "")),
        }

    if t == "label":
        mid = d.get("message_id")
        labels = d.get("labels") or []
        if not mid or not isinstance(labels, list):
            return {"action_type": "noop"}
        labs = [str(x) for x in labels]
        if not labs:
            return {"action_type": "noop"}
        return {
            "action_type": "label",
            "message_id": str(mid),
            "labels": labs,
            "replace_existing": bool(d.get("replace_existing", True)),
        }

    if t == "delete":
        mid = d.get("message_id")
        if not mid:
            return {"action_type": "noop"}
        return {
            "action_type": "delete",
            "message_id": str(mid),
            "permanent": bool(d.get("permanent", False)),
        }

    if t == "search":
        q = str(d.get("query", "")).strip()
        if not q:
            return {"action_type": "noop"}
        lim = d.get("limit", 20)
        try:
            lim_i = int(lim)
        except (TypeError, ValueError):
            lim_i = 20
        lim_i = max(1, min(500, lim_i))
        fs = d.get("folder_scope")
        return {
            "action_type": "search",
            "query": q,
            "folder_scope": fs,
            "limit": lim_i,
        }

    return {"action_type": "noop", "reason": d.get("reason")}


def _parse_llm_action(content: str) -> dict[str, Any]:
    parsed = _extract_json_object(content)
    if parsed is None:
        print("[warn] LLM output not valid JSON; using noop", file=sys.stderr)
        return {"action_type": "noop"}
    return _coerce_action(parsed)


def _fetch_task_limits(env_base: str, client: httpx.Client) -> dict[str, int]:
    r = client.get(f"{env_base.rstrip('/')}/tasks", timeout=30.0)
    r.raise_for_status()
    rows = r.json()
    out: dict[str, int] = {}
    for row in rows:
        out[str(row["id"])] = int(row["max_steps"])
    return out


def run_episode(
    task_id: str,
    env_base: str,
    http: httpx.Client,
    llm: OpenAI,
    model: str,
    max_steps: int,
    seed: int = 42,
    wall_deadline_monotonic: float | None = None,
) -> tuple[float, bool]:
    if wall_deadline_monotonic is not None and time.monotonic() >= wall_deadline_monotonic:
        return 0.0, False

    reset_r = http.post(
        f"{env_base.rstrip('/')}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=60.0,
    )
    reset_r.raise_for_status()
    reset_data = reset_r.json()
    session_id = reset_data["session_id"]
    obs = reset_data["observation"]

    last_step: dict[str, Any] | None = None
    final_holistic = float((obs.get("info") or {}).get("holistic_score", 0.0))
    done = False

    for i in range(max_steps):
        if wall_deadline_monotonic is not None and time.monotonic() >= wall_deadline_monotonic:
            break

        if obs.get("done"):
            done = True
            final_holistic = float((obs.get("info") or {}).get("holistic_score", 0.0))
            break

        user_prompt = _build_user_prompt(task_id, obs, i, max_steps, last_step)

        if wall_deadline_monotonic is not None and time.monotonic() >= wall_deadline_monotonic:
            break

        completion = llm.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=250,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        msg = completion.choices[0].message
        content = (msg.content or "").strip()
        action = _parse_llm_action(content)

        step_r = http.post(
            f"{env_base.rstrip('/')}/step",
            json={"session_id": session_id, "action": action},
            timeout=60.0,
        )
        step_r.raise_for_status()
        last_step = step_r.json()
        obs = last_step["observation"]
        final_holistic = float((last_step.get("info") or {}).get("holistic_score", 0.0))
        done = bool(last_step.get("done"))
        if done:
            break

    return final_holistic, done


def main() -> None:
    api_base = _env("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = _env("HF_TOKEN") or _env("API_KEY") or ""
    model = _env("MODEL_NAME") or ""
    env_base = _env("ENV_BASE_URL", "http://localhost:7860")
    wall_limit = float(_env("INFERENCE_WALL_LIMIT_SEC", "1140") or "1140")
    llm_timeout = float(_env("OPENAI_TIMEOUT_SEC", "45") or "45")
    seed_str = _env("INFERENCE_SEED", "42") or "42"
    episode_seed = int(seed_str)

    if not model:
        print("MODEL_NAME must be set", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("HF_TOKEN or API_KEY must be set for the API", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=api_base, api_key=api_key, timeout=llm_timeout)

    limits: dict[str, int] = {}
    results: list[tuple[str, float, bool]] = []
    wall_deadline = time.monotonic() + max(60.0, wall_limit)

    with httpx.Client(timeout=120.0) as http:
        limits = _fetch_task_limits(env_base, http)

        for task_id in TASK_ORDER:
            max_steps = limits.get(task_id, 30)
            print(f"\n=== Task: {task_id} (max_steps={max_steps}, seed={episode_seed}) ===", flush=True)
            try:
                score, done = run_episode(
                    task_id,
                    env_base,
                    http,
                    llm,
                    model,
                    max_steps,
                    seed=episode_seed,
                    wall_deadline_monotonic=wall_deadline,
                )
            except Exception as e:
                print(f"[error] {task_id}: {e}", file=sys.stderr)
                results.append((task_id, 0.0, False))
                continue
            results.append((task_id, score, done))
            print(f"Final holistic score: {score:.4f}  done={done}", flush=True)

    order_index = {tid: i for i, tid in enumerate(TASK_ORDER)}
    table_rows = sorted(results, key=lambda row: order_index.get(row[0], 999))

    print("\n" + "=" * 56)
    print(f"{'Task':<20} {'FinalScore':>12} {'Done':>8}")
    print("-" * 56)
    for tid, sc, d in table_rows:
        print(f"{tid:<20} {sc:>12.4f} {str(d):>8}")
    print("=" * 56)


if __name__ == "__main__":
    main()
