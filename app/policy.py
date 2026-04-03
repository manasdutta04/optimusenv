from __future__ import annotations

from typing import Any

from app.models import Action


def default_action_for_task(task_id: str) -> Action:
    defaults: dict[str, Action] = {
        "task_1": Action(
            learning_rate=0.0015,
            batch_size=64,
            weight_decay=1e-4,
            optimizer="adamw",
            num_layers=2,
            hidden_dim=128,
            use_amp=False,
            lr_schedule="cosine",
        ),
        "task_2": Action(
            learning_rate=0.001,
            batch_size=64,
            weight_decay=5e-4,
            optimizer="adamw",
            num_layers=3,
            hidden_dim=256,
            use_amp=False,
            lr_schedule="cosine",
        ),
        "task_3": Action(
            learning_rate=0.0012,
            batch_size=128,
            weight_decay=3e-4,
            optimizer="adamw",
            num_layers=2,
            hidden_dim=256,
            use_amp=False,
            lr_schedule="cosine",
        ),
    }
    return defaults.get(task_id, Action())


def heuristic_action_for_step(
    task_id: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
    max_epochs: int,
) -> Action:
    base = default_action_for_task(task_id)
    current_config = observation.get("current_config") or {}
    merged = {**base.model_dump(), **current_config}
    action = Action(**merged)
    step_index = len(history)

    # Avoid rebuilding the model mid-episode unless we are still at epoch 0.
    if history:
        action.num_layers = int(current_config.get("num_layers", action.num_layers))
        action.hidden_dim = int(current_config.get("hidden_dim", action.hidden_dim))

    if not history:
        return action

    prev = history[-1]
    val_loss = float(observation.get("val_loss", prev.get("val_loss", 0.0)))
    val_acc = float(observation.get("val_accuracy", prev.get("val_accuracy", 0.0)))
    prev_loss = float(prev.get("val_loss", val_loss))
    prev_acc = float(prev.get("val_accuracy", val_acc))
    throughput = float(observation.get("throughput_sps", prev.get("throughput_sps", 0.0)))

    if task_id == "task_1":
        if val_loss > prev_loss * 1.02:
            action.learning_rate = max(2e-4, action.learning_rate * 0.6)
            action.weight_decay = min(1e-3, max(action.weight_decay, 2e-4))
        if step_index >= max_epochs - 3:
            action.learning_rate = min(action.learning_rate, 7e-4)
        if val_acc > 0.955:
            action.batch_size = 128
        return action

    if task_id == "task_2":
        if val_loss > prev_loss * 1.02 and val_acc <= prev_acc + 0.005:
            action.learning_rate = max(2e-4, action.learning_rate * 0.7)
            action.weight_decay = min(3e-3, max(action.weight_decay, 1e-3))
        elif step_index >= max_epochs // 2:
            action.learning_rate = max(2e-4, action.learning_rate * 0.85)
        if throughput > 3500 and val_acc >= 0.84:
            action.batch_size = 128
        else:
            action.batch_size = 64
        return action

    if task_id == "task_3":
        if val_loss > prev_loss * 1.03:
            action.learning_rate = max(2e-4, action.learning_rate * 0.65)
            action.weight_decay = min(2e-3, max(action.weight_decay, 7e-4))
        elif step_index >= max_epochs // 2:
            action.learning_rate = max(2.5e-4, action.learning_rate * 0.85)
        if step_index >= 4 and val_acc < 0.42:
            action.batch_size = 64
        elif throughput > 1800:
            action.batch_size = 128
        return action

    return action
