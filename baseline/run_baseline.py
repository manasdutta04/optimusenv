#!/usr/bin/env python3
"""
OptimusEnv baseline agent - random policy.
Usage: python baseline/run_baseline.py [--host http://localhost:7860]
Must complete without errors and print scores for all 3 tasks.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any

import requests


def sample_action() -> dict[str, Any]:
    return {
        "learning_rate": 10 ** random.uniform(-4, -2),
        "batch_size": random.choice([32, 64, 128]),
        "weight_decay": random.choice([0.0, 0.0001, 0.001]),
        "optimizer": random.choice(["adam", "sgd", "adamw"]),
        "num_layers": random.randint(1, 4),
        "hidden_dim": random.choice([64, 128, 256]),
        "use_amp": False,
        "lr_schedule": random.choice(["none", "cosine"]),
    }


def run_task(host: str, task_id: str) -> tuple[float, int]:
    r = requests.post(f"{host}/reset", json={"task": task_id}, timeout=30)
    r.raise_for_status()

    done = False
    steps = 0
    while not done:
        action = sample_action()
        r = requests.post(f"{host}/step", json=action, timeout=30)
        r.raise_for_status()
        resp = r.json()
        done = bool(resp["done"])
        steps += 1

    r = requests.post(f"{host}/grader", timeout=30)
    r.raise_for_status()
    return float(r.json()["score"]), steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:7860")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("\nOptimusEnv Baseline Agent")
    print(f"Host: {args.host}")
    print(f"Seed: {args.seed}")
    print("-" * 40)

    scores: dict[str, float] = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        label = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}[task_id]
        print(f"Running {task_id} ({label})...", end=" ", flush=True)
        t0 = time.time()
        score, steps = run_task(args.host, task_id)
        elapsed = time.time() - t0
        scores[task_id] = score
        print(f"score={score:.4f}  steps={steps}  time={elapsed:.1f}s")

    avg = sum(scores.values()) / 3
    print("-" * 40)
    print(f"Task 1 (easy):    {scores['task_1']:.4f}")
    print(f"Task 2 (medium):  {scores['task_2']:.4f}")
    print(f"Task 3 (hard):    {scores['task_3']:.4f}")
    print(f"Average:          {avg:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
