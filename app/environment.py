from __future__ import annotations

from typing import Any, Optional

import numpy as np
from torch import nn

from app.models import Action, Observation, ResetResponse, StateResponse, StepResponse
from app.simulator import TrainingSimulator
from app.tasks import TASKS


class OptimusEnv:
    """Stateful environment. Global singleton, one episode at a time."""

    def __init__(self) -> None:
        self.simulator: Optional[TrainingSimulator] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.train_loader: Any = None
        self.val_loader: Any = None
        self.current_epoch: int = 0
        self.max_epochs: int = 0
        self.task_config: Optional[dict[str, Any]] = None
        self.episode_history: list[dict[str, Any]] = []
        self.prev_val_accuracy: float = 0.0
        self.episode_active: bool = False
        self.current_action: Optional[Action] = None

    def _apply_locked_params(self, action: Action) -> Action:
        if not self.task_config:
            return action

        payload = action.model_dump()
        locked = self.task_config.get("locked_params", [])
        fixed_arch = self.task_config.get("fixed_arch", {})
        for key in locked:
            if key in fixed_arch:
                payload[key] = fixed_arch[key]
        return Action(**payload)

    def _build_runtime(self, action: Action) -> None:
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized")

        self.model = self.simulator.build_model(action)
        self.optimizer = self.simulator.build_optimizer(self.model, action)
        self.scheduler = self.simulator.build_scheduler(
            self.optimizer, action, self.max_epochs
        )
        self.train_loader, self.val_loader = self.simulator.create_dataloaders(
            action.batch_size
        )

    def reset(self, task_id: str = "task_1") -> ResetResponse:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task_config = dict(TASKS[task_id])
        self.simulator = TrainingSimulator(self.task_config)

        self.current_epoch = 0
        self.max_epochs = int(self.task_config["max_epochs"])
        self.episode_history = []
        self.prev_val_accuracy = 0.0
        self.episode_active = True

        default_action = self._apply_locked_params(Action())
        self.current_action = default_action
        self._build_runtime(default_action)

        observation = Observation(
            epoch=0,
            train_loss=0.0,
            val_loss=0.0,
            val_accuracy=0.0,
            throughput_sps=0.0,
            mem_mb=0.0,
            time_elapsed_sec=0.0,
            current_config=default_action.model_dump(),
        )
        return ResetResponse(
            observation=observation,
            task=self.task_config["task_id"],
            task_description=self.task_config["description"],
            max_epochs=self.max_epochs,
            action_schema=Action.model_json_schema(),
        )

    def step(self, action: Action) -> StepResponse:
        if not self.episode_active:
            raise ValueError("No active episode. Call reset() first.")
        if self.simulator is None or self.model is None:
            raise RuntimeError("Environment is not initialized.")
        if self.current_action is None:
            raise RuntimeError("Current action is missing.")

        effective_action = self._apply_locked_params(action)
        previous_action = self.current_action

        architecture_changed = (
            effective_action.num_layers != previous_action.num_layers
            or effective_action.hidden_dim != previous_action.hidden_dim
        )
        batch_size_changed = effective_action.batch_size != previous_action.batch_size
        optimizer_changed = (
            effective_action.optimizer != previous_action.optimizer
            or effective_action.learning_rate != previous_action.learning_rate
            or effective_action.weight_decay != previous_action.weight_decay
        )
        scheduler_changed = effective_action.lr_schedule != previous_action.lr_schedule

        if architecture_changed:
            self.model = self.simulator.build_model(effective_action)
            self.optimizer = self.simulator.build_optimizer(self.model, effective_action)
            self.scheduler = self.simulator.build_scheduler(
                self.optimizer, effective_action, self.max_epochs
            )
        elif optimizer_changed:
            self.optimizer = self.simulator.build_optimizer(self.model, effective_action)
            self.scheduler = self.simulator.build_scheduler(
                self.optimizer, effective_action, self.max_epochs
            )
        elif scheduler_changed:
            self.scheduler = self.simulator.build_scheduler(
                self.optimizer, effective_action, self.max_epochs
            )

        if batch_size_changed or self.train_loader is None or self.val_loader is None:
            self.train_loader, self.val_loader = self.simulator.create_dataloaders(
                effective_action.batch_size
            )

        metrics = self.simulator.run_epoch(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            action=effective_action,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )

        next_epoch = self.current_epoch + 1
        crashed = bool(metrics.get("crashed", False))

        if crashed:
            observation = Observation(
                epoch=next_epoch,
                train_loss=0.0,
                val_loss=0.0,
                val_accuracy=0.0,
                throughput_sps=0.0,
                mem_mb=0.0,
                time_elapsed_sec=0.0,
                current_config=effective_action.model_dump(),
            )
            reward = self.compute_reward(
                prev_acc=self.prev_val_accuracy,
                curr_obs=observation,
                crashed=True,
            )
            info: dict[str, Any] = {
                "crashed": True,
                "reason": metrics.get("reason", "unknown"),
            }
        else:
            observation = Observation(
                epoch=next_epoch,
                train_loss=float(metrics["train_loss"]),
                val_loss=float(metrics["val_loss"]),
                val_accuracy=float(metrics["val_accuracy"]),
                throughput_sps=float(metrics["throughput_sps"]),
                mem_mb=float(metrics["mem_mb"]),
                time_elapsed_sec=float(metrics["time_elapsed_sec"]),
                current_config=effective_action.model_dump(),
            )
            reward = self.compute_reward(
                prev_acc=self.prev_val_accuracy,
                curr_obs=observation,
                crashed=False,
            )
            self.prev_val_accuracy = observation.val_accuracy
            info = {
                "crashed": False,
                "reason": None,
            }

        self.episode_history.append(observation.model_dump())
        self.current_epoch = next_epoch
        self.current_action = effective_action

        done = crashed or self.current_epoch >= self.max_epochs
        if done:
            self.episode_active = False

        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def get_state(self) -> StateResponse:
        return StateResponse(
            episode_active=self.episode_active,
            task=self.task_config["task_id"] if self.task_config else None,
            epoch=self.current_epoch,
            max_epochs=self.max_epochs,
            steps_taken=len(self.episode_history),
            latest_observation=self.episode_history[-1] if self.episode_history else None,
        )

    def compute_reward(
        self,
        prev_acc: float,
        curr_obs: Observation,
        crashed: bool = False,
    ) -> float:
        if crashed:
            return -1.0

        acc_delta = curr_obs.val_accuracy - prev_acc
        acc_reward = acc_delta * 2.0

        speed_bonus = max(0.0, (curr_obs.throughput_sps / 500.0 - 1.0)) * 0.15
        mem_bonus = max(0.0, (1.0 - curr_obs.mem_mb / 512.0)) * 0.10

        total = acc_reward + speed_bonus + mem_bonus
        return float(np.clip(total, -1.0, 1.0))
