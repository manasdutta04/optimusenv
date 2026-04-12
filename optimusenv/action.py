from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class OptimusEnvAction(BaseModel):
    learning_rate: float = Field(default=0.001, ge=1e-5, le=0.1)
    batch_size: int = Field(default=64, ge=8, le=512)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=0.1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adamw"
    num_layers: int = Field(default=2, ge=1, le=5)
    hidden_dim: int = Field(default=128, ge=32, le=512)
    use_amp: bool = False
    lr_schedule: Literal["none", "cosine", "step"] = "none"
