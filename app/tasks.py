from __future__ import annotations

from typing import Any

TASK_1: dict[str, Any] = {
    "task_id": "task_1",
    "difficulty": "easy",
    "description": "Tune hyperparameters to maximize accuracy on MNIST.",
    "dataset": "mnist",
    "max_epochs": 10,
    "target_accuracy": 0.92,
    "locked_params": ["num_layers", "hidden_dim"],
    "fixed_arch": {"num_layers": 2, "hidden_dim": 128},
    "grader": "accuracy_grader",
}

TASK_2: dict[str, Any] = {
    "task_id": "task_2",
    "difficulty": "medium",
    "description": "Design architecture and tune training to generalize across MNIST and FashionMNIST.",
    "datasets": ["mnist", "fashion_mnist"],
    "max_epochs": 15,
    "target_accuracy": 0.88,
    "locked_params": [],
    "grader": "generalization_grader",
}

TASK_3: dict[str, Any] = {
    "task_id": "task_3",
    "difficulty": "hard",
    "description": "Maximize accuracy AND training efficiency (speed + memory) on CIFAR-10.",
    "dataset": "cifar10",
    "max_epochs": 20,
    "target_accuracy": 0.60,
    "locked_params": [],
    "grader": "composite_grader",
}

TASKS: dict[str, dict[str, Any]] = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}
