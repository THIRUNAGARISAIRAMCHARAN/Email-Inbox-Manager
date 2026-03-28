# Email Inbox Manager (OpenEnv)

## 1. Overview

**Email Inbox Manager** is an OpenEnv-style benchmark in which an **AI agent operates a simulated email inbox**: it reads messages, replies, composes mail, applies labels, deletes messages, and searches—under explicit task instructions and step budgets. The environment is **deterministic given a PRNG seed**, exposes **typed observations and actions** over **HTTP**, and scores behavior with **dense, interpretable rewards** suitable for evaluating tool use, planning, and instruction following in a **realistic productivity domain** without live mail systems or third-party APIs during rollouts.

---

## 2. Environment description

| Aspect | Description |
|--------|-------------|
| **Inbox simulation** | Stateful list of `EmailMessage` rows (subject, body, participants, read flags, folders, thread metadata). Actions mutate this list, append **sent** messages, and update search views. |
| **State** | Full snapshot is `InboxState` (emails, optional focus id, `TaskSpec`, step counter, `done`, trash ids, folder index, search fields, cumulative reward, sent messages). HTTP access via `GET /state/{session_id}`. |
| **Email generation** | `email_env.data_gen.generate_inbox(task_id, seed)` builds reproducible inboxes: **easy_reply** (boss deadline mail among fillers), **medium_triage** (newsletters, `needs_reply`, spam—all unread), **hard_thread** (four-message Q4 budget thread plus noise). No external data libraries. |
| **Episode lifecycle** | `POST /reset` creates a new `session_id`, seeds `EmailEnv`, returns `ResetResponse` with initial `Observation`. `POST /step` applies one discriminated `Action`. Terminal when **holistic score ≥ 1.0** or **step count ≥ max_steps** for the task. |

---

## 3. Action space

Discriminated union on **`action_type`** (`email_env.models.Action`). JSON uses **snake_case**; **`message_id`** identifies messages.

| `action_type` | Parameters | Semantics |
|---------------|------------|-----------|
| **read_email** | `message_id` (str), `mark_read` (bool, default `true`) | Open mail; optionally mark read. |
| **reply** | `message_id`, `body`, optional `reply_all`, `cc_addresses` | Send a reply tied to parent; recorded in `sent_messages`. |
| **compose** | optional `draft_id`; `to_addresses`, `cc_addresses`, `bcc_addresses`; `subject`, `body` | New outbound message. |
| **label** | `message_id`, `labels` (list[str]), `replace_existing` (bool) | Set folder / labels on a message. |
| **delete** | `message_id`, `permanent` (bool) | Remove from inbox (soft trash unless permanent). Protected focus ids may be blocked. |
| **search** | `query`, optional `folder_scope`, `limit` | Substring match on subject/body/from; may narrow `inbox_snapshot` for that step. |
| **noop** | optional `reason` | No state change; holistic score unchanged unless other factors already moved. |

---

## 4. Observation space

Returned on **reset** (inside `ResetResponse.observation`) and inside **`StepResult.observation`** after each step.

| Field | Type | Role |
|-------|------|------|
| `task_id` | string | Active task key. |
| `task_type` | string | Same as `task_id` in this build. |
| `prompt` | string | Instruction text from `TaskConfig.desc`. |
| `inbox_snapshot` | `EmailMessage[]` | Visible messages (may be filtered immediately after **search**). |
| `focus_message_id` | string \| null | Emphasized message when set. |
| `done` | boolean | Episode terminated. |
| `reward` | number \| null | Transition-level signal (e.g. step delta context). |
| `step_count` | integer | Steps taken so far (≥ 0). |
| `schema_version` | string | Observation schema revision. |
| `available_action_types` | string[] | Allowed `action_type` discriminators. |
| `standard_folders` | string[] | Triage folders (e.g. inbox, updates, finance, social). |
| `info` | object | **Diagnostics only** (`holistic_score`, `cumulative_reward`, `max_steps`, search hits, etc.); **no private grader gold.** |

Each **EmailMessage** includes: `message_id`, `subject`, `body`, `from_address`, `to_addresses`, `cc_addresses`, `bcc_addresses`, `date_sent`, `read`, `thread_id`, `in_reply_to`, `folder`, `importance`, `has_attachments`, `snippet`, `metadata`.

---

## 5. Tasks

| Task id | Name | Difficulty | max_steps | Success criteria (holistic = **1.0**) |
|---------|------|------------|-----------|----------------------------------------|
| `easy_reply` | Boss urgent reply | easy | 10 | Reply **to the boss message** (`boss@company.com`) with **body length > 20** (strip whitespace). *(0.5 if boss mail read only; 0.0 otherwise.)* |
| `medium_triage` | Inbox triage | medium | 20 | **Weighted completion**: 0.33×fraction of **newsletters** filed as **updates** (or labeled updates/newsletter) + 0.33×fraction of **`needs_reply`** messages receiving a **reply** + 0.34×fraction of **spam** removed from inbox. Empty class counts as fully satisfied for that factor. |
| `hard_thread` | Q4 budget thread | hard | 30 | **Reply or compose** with **body > 50** chars and text mentioning **q4**, **budget**, or **meeting** (case-insensitive). *(Intermediate: 0.5 if all thread messages read and no qualifying send; 0.2 if any thread message read.)* |

Episode **success** is commonly taken as **final holistic score = 1.0** before or at max steps.

---

## 6. Reward function

- After every step the environment recomputes a **holistic score** *h* ∈ [0,1] via **`get_grader(task_id)(GraderContext)`** (deterministic from inbox + sent + initial snapshot).
- **`StepResult.reward`** is the **step-wise delta**: Δ*h* = *h*<sub>new</sub> − *h*<sub>prev</sub> (clamped for API typing), yielding a **dense** signal whenever progress increases.
- **`Observation` / `StepResult.info`** exposes **`holistic_score`** and **`cumulative_reward`** (sum of per-step deltas) for logging.
- **Partial credit** is explicit: **easy** has a **0.5** read-only tier; **medium** decomposes into three weighted fractions; **hard** uses **0.2 / 0.5 / 1.0** tiers before a qualifying outbound message.
- Episode ends when **h = 1.0** (success) or **steps ≥ max_steps** (timeout).

---

## 7. Setup instructions

### Local (pip + uvicorn)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Requires **Python 3.11+** (Dockerfile targets 3.11).

### Docker

```bash
docker build -t email-inbox-manager .
docker run --rm -p 7860:7860 email-inbox-manager
```

Image listens on **7860**; **HEALTHCHECK** hits **`GET /health`**.

---

## 8. Running inference

1. Start the API (see **§7**).
2. Set LLM router / OpenAI-compatible variables, then run:

| Variable | Required | Default |
|----------|----------|---------|
| `MODEL_NAME` | Yes | — |
| `HF_TOKEN` or `API_KEY` | Yes | — |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` |
| `ENV_BASE_URL` | No | `http://localhost:7860` |

```bash
export MODEL_NAME="your-model-id"
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"   # optional
python inference.py
```

The script loops **`easy_reply` → `medium_triage` → `hard_thread`**, calls **`POST /reset`** and **`POST /step`** via **httpx**, and queries the LLM with the **OpenAI** client. It prints per-task scores and a **final summary table**.

---

## 9. OpenEnv compliance

- **Typed contract**: Pydantic v2 models for **Observation**, **Action** (discriminated union), **InboxState**, **ResetResponse**, **StepResult**.
- **Stateful episodes**: `EmailEnv` + **`SESSION_STORE`** keyed by **`session_id`**; thread-safe locks per session.
- **Core API surface**: **`POST /reset`**, **`POST /step`**, **`GET /state/{session_id}`**, **`GET /tasks`**, **`GET /health`**.
- **Task registry**: **`TASK_REGISTRY`** / **`TaskConfig`** (`id`, difficulty, `max_steps`, description).
- **Manifest**: **`openenv.yaml`** documents metadata, API, observation/action space, tasks, reward intent, Docker.
- **Reproducibility**: **`seed`** on reset + deterministic **`generate_inbox`** and graders.
- **Containerization**: Production-oriented **Dockerfile** (slim base, layer caching, non-root user, healthcheck, port **7860**).

---

## 10. Baseline scores (placeholder)

Representative **mean holistic score at max_steps** (fill in after official eval runs):

| Task | Random policy | Heuristic baseline | Strong LLM (TBD) |
|------|----------------|-------------------|------------------|
| `easy_reply` | *TBD* | *TBD* | *TBD* |
| `medium_triage` | *TBD* | *TBD* | *TBD* |
| `hard_thread` | *TBD* | *TBD* | *TBD* |

Replace *TBD* with aggregate metrics (mean, std, success@1.0) once benchmarks are frozen.

---

## Project layout

| Path | Role |
|------|------|
| `main.py` | FastAPI app, CORS, lifespan, session store |
| `email_env/env.py` | `EmailEnv`: reset, step, state |
| `email_env/models.py` | Pydantic models |
| `email_env/tasks.py` | `TASK_REGISTRY`, `TaskConfig` |
| `email_env/graders.py` | `GraderContext`, `get_grader` |
| `email_env/data_gen.py` | `generate_inbox` |
| `inference.py` | LLM + httpx evaluation loop |
| `openenv.yaml` | OpenEnv manifest |
| `Dockerfile` / `requirements.txt` | Deploy and dependencies |

*Environment name: **email-inbox-manager** — version aligned with `openenv.yaml`.*
