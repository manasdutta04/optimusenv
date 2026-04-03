#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any

import requests

from app.models import Action
from app.policy import heuristic_action_for_step


SYSTEM_PROMPT = """You are an expert ML engineer optimizing a real PyTorch training run.
Return a single JSON object matching this schema:
{
  "learning_rate": float,
  "batch_size": int,
  "weight_decay": float,
  "optimizer": "adam" | "sgd" | "adamw",
  "num_layers": int,
  "hidden_dim": int,
  "use_amp": false,
  "lr_schedule": "none" | "cosine" | "step"
}
Keep architecture stable after the first epoch unless the current setup is clearly failing.
Prefer small, sensible adjustments over drastic jumps."""


def create_openai_client() -> Any | None:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


def call_llm_for_action(
    client: Any,
    model_name: str,
    task_id: str,
    task_description: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
    max_epochs: int,
) -> Action | None:
    heuristic = heuristic_action_for_step(task_id, observation, history, max_epochs)
    prompt = {
        "task_id": task_id,
        "task_description": task_description,
        "max_epochs": max_epochs,
        "history_tail": history[-4:],
        "current_observation": observation,
        "heuristic_suggestion": heuristic.model_dump(),
        "instruction": "Return only the next action as JSON.",
    }
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
        return Action.model_validate_json(content)
    except Exception:
        return None


def should_query_llm(history: list[dict[str, Any]], max_epochs: int) -> bool:
    if not history:
        return True
    if len(history) == max_epochs // 2:
        return True
    if len(history) < 2:
        return False
    latest = history[-1]
    previous = history[-2]
    return (
        float(latest["val_loss"]) > float(previous["val_loss"]) * 1.04
        and float(latest["val_accuracy"]) <= float(previous["val_accuracy"]) + 0.003
    )


def run_task(host: str, task_id: str, client: Any | None, model_name: str) -> tuple[float, int]:
    reset_response = requests.post(f"{host}/reset", json={"task": task_id}, timeout=60)
    reset_response.raise_for_status()
    reset_payload = reset_response.json()
    task_description = reset_payload["task_description"]
    max_epochs = int(reset_payload["max_epochs"])
    observation = reset_payload["observation"]
    history: list[dict[str, Any]] = []
    done = False
    steps = 0

    while not done:
        action = heuristic_action_for_step(task_id, observation, history, max_epochs)
        if client is not None and should_query_llm(history, max_epochs):
            llm_action = call_llm_for_action(
                client=client,
                model_name=model_name,
                task_id=task_id,
                task_description=task_description,
                observation=observation,
                history=history,
                max_epochs=max_epochs,
            )
            if llm_action is not None:
                action = llm_action

        response = requests.post(
            f"{host}/step",
            json=action.model_dump(),
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        observation = payload["observation"]
        history.append(observation)
        done = bool(payload["done"])
        steps += 1

    grader_response = requests.post(f"{host}/grader", timeout=60)
    grader_response.raise_for_status()
    return float(grader_response.json()["score"]), steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon inference for OptimusEnv")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    args = parser.parse_args()

    random.seed(args.seed)
    client = create_openai_client()

    print("OptimusEnv Inference")
    print(f"Host: {args.host}")
    print(f"Seed: {args.seed}")
    print(f"Model: {args.model}")
    print(f"LLM enabled: {'yes' if client is not None else 'no (heuristic fallback)'}")
    print("-" * 40)

    scores: dict[str, float] = {}
    start = time.time()
    for task_id in ["task_1", "task_2", "task_3"]:
        task_start = time.time()
        score, steps = run_task(args.host, task_id, client, args.model)
        elapsed = time.time() - task_start
        scores[task_id] = score
        print(f"{task_id}: score={score:.4f} steps={steps} time={elapsed:.1f}s")

    average = sum(scores.values()) / 3.0
    total_elapsed = time.time() - start
    print("-" * 40)
    print(json.dumps({**scores, "average": round(average, 4), "total_time_sec": round(total_elapsed, 1)}, indent=2))


if __name__ == "__main__":
    main()
