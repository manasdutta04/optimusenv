from __future__ import annotations

from typing import Any, Callable

import numpy as np

from app.models import GraderResponse


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def accuracy_grader(
    episode_history: list[dict[str, Any]],
    task_config: dict[str, Any],
) -> GraderResponse:
    best_acc = max((float(obs["val_accuracy"]) for obs in episode_history), default=0.0)
    target = float(task_config.get("target_accuracy", 0.92))
    score = _clamp01(min(best_acc / max(target, 1e-8), 1.0))
    return GraderResponse(
        score=score,
        task=str(task_config["task_id"]),
        breakdown={
            "best_val_accuracy": float(best_acc),
            "target": float(target),
            "epochs_run": len(episode_history),
        },
    )


def generalization_grader(
    episode_history: list[dict[str, Any]],
    task_config: dict[str, Any],
) -> GraderResponse:
    final_acc = float(episode_history[-1]["val_accuracy"]) if episode_history else 0.0
    target = float(task_config.get("target_accuracy", 0.88))
    base_score = min(final_acc / max(target, 1e-8), 1.0)
    epochs = len(episode_history)
    if epochs < 10:
        base_score *= epochs / 10.0
    score = _clamp01(base_score)
    return GraderResponse(
        score=score,
        task=str(task_config["task_id"]),
        breakdown={
            "final_accuracy": float(final_acc),
            "epochs_completed": epochs,
            "target": float(target),
        },
    )


def composite_grader(
    episode_history: list[dict[str, Any]],
    task_config: dict[str, Any],
) -> GraderResponse:
    best_accuracy = max((float(obs["val_accuracy"]) for obs in episode_history), default=0.0)
    best_throughput = max(
        (float(obs["throughput_sps"]) for obs in episode_history), default=0.0
    )
    min_memory_mb = min((float(obs["mem_mb"]) for obs in episode_history), default=1024.0)

    target = float(task_config.get("target_accuracy", 0.60))
    accuracy_score = _clamp01(min(best_accuracy / max(target, 1e-8), 1.0))
    speed_score = _clamp01(min(best_throughput / 1000.0, 1.0))
    memory_score = _clamp01(max(0.0, 1.0 - min_memory_mb / 1024.0))

    final_score = _clamp01(
        accuracy_score * 0.5 + speed_score * 0.3 + memory_score * 0.2
    )

    return GraderResponse(
        score=final_score,
        task=str(task_config["task_id"]),
        breakdown={
            "accuracy_score": float(accuracy_score),
            "speed_score": float(speed_score),
            "memory_score": float(memory_score),
            "best_accuracy": float(best_accuracy),
            "best_throughput": float(best_throughput),
            "min_memory_mb": float(min_memory_mb),
        },
    )


def get_grader(name: str) -> Callable[[list[dict[str, Any]], dict[str, Any]], GraderResponse]:
    graders: dict[str, Callable[[list[dict[str, Any]], dict[str, Any]], GraderResponse]] = {
        "accuracy_grader": accuracy_grader,
        "generalization_grader": generalization_grader,
        "composite_grader": composite_grader,
    }
    if name not in graders:
        raise ValueError(f"Unknown grader: {name}")
    return graders[name]
