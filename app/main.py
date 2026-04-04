from __future__ import annotations

import os
import random
import threading
from contextlib import contextmanager

import torch
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.environment import OptimusEnv
from app.graders import get_grader
from app.models import (
    Action,
    BaselineResponse,
    GraderResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
    TaskInfo,
)
from app.policy import heuristic_action_for_step
from app.simulator import TrainingSimulator
from app.tasks import TASKS

app = FastAPI(
    title="OptimusEnv",
    version="1.0.0",
    description="RL environment for PyTorch training optimization",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/web", StaticFiles(directory=static_dir, html=True), name="web")

@app.get("/")
def read_root():
    return RedirectResponse(url="/web")

# Global singleton
env = OptimusEnv()
_env_lock = threading.Lock()


@contextmanager
def exclusive_env_access():
    if not _env_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Environment is busy with another episode. Retry after the current run finishes.",
        )
    try:
        yield
    finally:
        _env_lock.release()


@app.on_event("startup")
async def startup() -> None:
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("OptimusEnv starting - pre-caching datasets...")
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            _ = TrainingSimulator(TASKS[task_id])
            print(f"Dataset cache ready for {task_id}")
        except Exception as exc:
            print(f"Dataset preload warning for {task_id}: {exc}")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", env="OptimusEnv", version="1.0.0")


@app.post("/reset")
def reset(body: ResetRequest = Body(default=ResetRequest(task="task_1"))) -> ResetResponse:
    task_id = body.task
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")
    try:
        with exclusive_env_access():
            return env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action) -> StepResponse:
    if not env.episode_active:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    try:
        with exclusive_env_access():
            return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state() -> StateResponse:
    try:
        return env.get_state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tasks")
def tasks() -> list[TaskInfo]:
    try:
        action_schema = Action.model_json_schema()
        return [
            TaskInfo(
                task_id=t["task_id"],
                description=t["description"],
                difficulty=t["difficulty"],
                max_epochs=t["max_epochs"],
                action_schema=action_schema,
            )
            for t in TASKS.values()
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/grader")
def grader() -> GraderResponse:
    if env.episode_active:
        raise HTTPException(status_code=400, detail="Episode still active. Complete it first.")
    if not env.episode_history:
        raise HTTPException(status_code=400, detail="No episode history. Run an episode first.")
    if env.task_config is None:
        raise HTTPException(status_code=400, detail="No active task configuration.")
    try:
        with exclusive_env_access():
            grader_fn = get_grader(env.task_config["grader"])
            return grader_fn(env.episode_history, env.task_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/baseline")
def baseline() -> BaselineResponse:
    scores: dict[str, float] = {}
    try:
        with exclusive_env_access():
            for task_id in ["task_1", "task_2", "task_3"]:
                reset_response = env.reset(task_id)
                done = False
                observation = reset_response.observation.model_dump()
                history: list[dict[str, float | int | dict[str, Any]]] = []
                while not done:
                    action = heuristic_action_for_step(
                        task_id=task_id,
                        observation=observation,
                        history=history,
                        max_epochs=env.max_epochs,
                    )
                    resp = env.step(action)
                    done = resp.done
                    observation = resp.observation.model_dump()
                    history.append(observation)

                if env.task_config is None:
                    raise RuntimeError("Task config missing after episode.")
                grader_resp = get_grader(env.task_config["grader"])(
                    env.episode_history,
                    env.task_config,
                )
                scores[task_id] = round(float(grader_resp.score), 4)

            avg = sum(scores.values()) / 3.0
            scores["average"] = round(float(avg), 4)
            return BaselineResponse(**scores)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def sample_random_action() -> Action:
    return Action(
        learning_rate=10 ** random.uniform(-4, -2),
        batch_size=random.choice([32, 64, 128]),
        weight_decay=random.choice([0.0, 1e-4, 1e-3]),
        optimizer=random.choice(["adam", "sgd", "adamw"]),
        num_layers=random.randint(1, 4),
        hidden_dim=random.choice([64, 128, 256]),
        use_amp=False,
        lr_schedule=random.choice(["none", "cosine"]),
    )
