"""
FastAPI OpenEnv server: Email Inbox Manager.

Run: ``uvicorn main:app --host 0.0.0.0 --port 7860``
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from email_env.env import EmailEnv
from email_env.models import Action, InboxState, ResetResponse, StepResult
from email_env.tasks import TASK_REGISTRY, TaskConfig

# Process-local session table: session_id -> EmailEnv
SESSION_STORE: dict[str, EmailEnv] = {}
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_MAP_LOCK = threading.Lock()


def _lock_for_session(session_id: str) -> threading.RLock:
    with _MAP_LOCK:
        if session_id not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_id] = threading.RLock()
        return _SESSION_LOCKS[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Email Inbox Manager OpenEnv ready")
    yield


app = FastAPI(
    title="Email Inbox Manager",
    version="0.1.0",
    lifespan=lifespan,
)

# Wildcard origins require allow_credentials=False (browser / Starlette reject credentials + "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        default="easy_reply",
        description="Task id from TASK_REGISTRY (e.g. easy_reply). Unknown ids fall back to easy_reply.",
    )
    task_type: str | None = Field(
        default=None,
        description="Alias for task_id (OpenEnv compatibility); if set, overrides task_id.",
    )
    seed: int = Field(default=42, description="PRNG seed for synthetic inbox generation.")

    def effective_task_id(self) -> str:
        if self.task_type is not None and str(self.task_type).strip() != "":
            return str(self.task_type).strip()
        tid = str(self.task_id).strip()
        return tid if tid else "easy_reply"


class StepBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session id returned by POST /reset.")
    action: Action


@app.get("/")
def root():
    return {"name": "email-inbox-manager", "status": "ok", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "env": "email-inbox-manager"}


@app.post("/reset", response_model=ResetResponse)
def reset_episode(body: ResetBody):
    task_id = body.effective_task_id()
    if task_id not in TASK_REGISTRY:
        task_id = "easy_reply"

    session_id = uuid.uuid4().hex
    env = EmailEnv(task_id=task_id, session_id=session_id, seed=body.seed)

    with _MAP_LOCK:
        SESSION_STORE[session_id] = env
    lock = _lock_for_session(session_id)
    with lock:
        return env.reset()


@app.post("/step", response_model=StepResult)
def step_episode(body: StepBody):
    with _MAP_LOCK:
        env = SESSION_STORE.get(body.session_id)

    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {body.session_id!r}")

    lock = _lock_for_session(body.session_id)
    with lock:
        return env.step(body.action)


@app.get("/state/{session_id}", response_model=InboxState)
def state_session(session_id: str):
    with _MAP_LOCK:
        env = SESSION_STORE.get(session_id)

    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")

    lock = _lock_for_session(session_id)
    with lock:
        return env.state()


@app.get("/tasks", response_model=list[TaskConfig])
def tasks():
    return sorted(TASK_REGISTRY.values(), key=lambda c: c.id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)

