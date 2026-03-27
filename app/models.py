from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Action(BaseModel):
    learning_rate: float = Field(default=0.001, ge=1e-5, le=0.1)
    batch_size: int = Field(default=64, ge=8, le=512)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=0.1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adamw"
    num_layers: int = Field(default=2, ge=1, le=5)
    hidden_dim: int = Field(default=128, ge=32, le=512)
    use_amp: bool = False
    lr_schedule: Literal["none", "cosine", "step"] = "none"


class Observation(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    throughput_sps: float
    mem_mb: float
    time_elapsed_sec: float
    current_config: dict[str, Any]


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class ResetResponse(BaseModel):
    observation: Observation
    task: str
    task_description: str
    max_epochs: int
    action_schema: dict[str, Any]


class GraderResponse(BaseModel):
    score: float
    task: str
    breakdown: dict[str, Any]


class TaskInfo(BaseModel):
    task_id: str
    description: str
    difficulty: str
    max_epochs: int
    action_schema: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    env: str
    version: str


class ResetRequest(BaseModel):
    task: str = "task_1"


class StateResponse(BaseModel):
    episode_active: bool
    task: str | None
    epoch: int
    max_epochs: int
    steps_taken: int
    latest_observation: dict[str, Any] | None


class BaselineResponse(BaseModel):
    task_1: float
    task_2: float
    task_3: float
    average: float
