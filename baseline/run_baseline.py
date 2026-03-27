#!/usr/bin/env python3
"""
OptimusEnv baseline agent — LLM-powered (OpenAI API).

Uses an LLM to make intelligent hyperparameter decisions based on
training metrics feedback. Falls back to random actions if no API key.

Usage:
    # LLM agent (requires OPENAI_API_KEY env var):
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py

    # Random fallback:
    python baseline/run_baseline.py --host http://localhost:7860

    # Custom model:
    python baseline/run_baseline.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# LLM Agent (OpenAI API)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ML engineer optimizing a PyTorch training run.
You receive the current training metrics and must decide the next step's hyperparameters.

Guidelines:
- Start with moderate learning rates (0.001-0.01) and increase/decrease based on loss trends.
- If val_loss is decreasing, keep current config or make small adjustments.
- If val_loss is increasing (overfitting), increase weight_decay or reduce learning_rate.
- If training is slow, try larger batch_size.
- SGD with momentum can be more stable; Adam/AdamW often converges faster.
- Cosine LR schedule often helps in later epochs.
- More layers = more capacity but risk of overfitting on small datasets.

Return ONLY a valid JSON object with these fields (all required):
{
    "learning_rate": <float 1e-5 to 0.1>,
    "batch_size": <int 8 to 512, prefer 32/64/128>,
    "weight_decay": <float 0.0 to 0.1>,
    "optimizer": <"adam" | "sgd" | "adamw">,
    "num_layers": <int 1 to 5>,
    "hidden_dim": <int 32 to 512, prefer 64/128/256>,
    "use_amp": false,
    "lr_schedule": <"none" | "cosine" | "step">
}

No explanation, no markdown, just the JSON object."""


def create_openai_client() -> Optional[Any]:
    """Create OpenAI client if API key is available.
    
    Supports custom base URL via OPENAI_BASE_URL env var,
    enabling Ollama, vLLM, or any OpenAI-compatible server.
    
    Examples:
        # OpenAI:
        OPENAI_API_KEY=sk-... python baseline/run_baseline.py
        
        # Ollama (local):
        OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 python baseline/run_baseline.py --model llama3
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    except ImportError:
        print("WARNING: openai package not installed. Falling back to random agent.")
        return None


def llm_pick_action(
    client: Any,
    model: str,
    task_description: str,
    observation: dict[str, Any],
    epoch: int,
    max_epochs: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the LLM to pick the next action based on current metrics."""
    history_text = ""
    for h in history[-5:]:  # Last 5 observations for context
        history_text += (
            f"  Epoch {h['epoch']}: loss={h['train_loss']:.4f} "
            f"val_loss={h['val_loss']:.4f} val_acc={h['val_accuracy']:.4f} "
            f"throughput={h['throughput_sps']:.0f} sps\n"
        )

    user_msg = f"""Task: {task_description}
Current epoch: {epoch}/{max_epochs}

Current observation:
- train_loss: {observation.get('train_loss', 0):.4f}
- val_loss: {observation.get('val_loss', 0):.4f}
- val_accuracy: {observation.get('val_accuracy', 0):.4f}
- throughput: {observation.get('throughput_sps', 0):.0f} samples/sec
- memory: {observation.get('mem_mb', 0):.1f} MB

Current config: {json.dumps(observation.get('current_config', {}), indent=2)}

Recent history:
{history_text if history_text else '  (first epoch)'}

Pick the optimal hyperparameters for the next training epoch."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        action = json.loads(content)
        # Validate and clamp values
        action["learning_rate"] = max(1e-5, min(0.1, float(action.get("learning_rate", 0.001))))
        action["batch_size"] = max(8, min(512, int(action.get("batch_size", 64))))
        action["weight_decay"] = max(0.0, min(0.1, float(action.get("weight_decay", 1e-4))))
        action["optimizer"] = action.get("optimizer", "adamw") if action.get("optimizer") in ("adam", "sgd", "adamw") else "adamw"
        action["num_layers"] = max(1, min(5, int(action.get("num_layers", 2))))
        action["hidden_dim"] = max(32, min(512, int(action.get("hidden_dim", 128))))
        action["use_amp"] = bool(action.get("use_amp", False))
        action["lr_schedule"] = action.get("lr_schedule", "none") if action.get("lr_schedule") in ("none", "cosine", "step") else "none"
        return action
    except Exception as exc:
        print(f"    LLM error: {exc}. Using random action.")
        return sample_random_action()


# ---------------------------------------------------------------------------
# Random Agent (fallback)
# ---------------------------------------------------------------------------


def sample_random_action() -> dict[str, Any]:
    """Sample a random action for the environment."""
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


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_task(
    host: str,
    task_id: str,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
) -> tuple[float, int]:
    """Run a full episode for a task and return (grader_score, steps)."""
    r = requests.post(f"{host}/reset", json={"task": task_id}, timeout=60)
    r.raise_for_status()
    reset_data = r.json()
    task_description = reset_data["task_description"]
    max_epochs = reset_data["max_epochs"]
    observation = reset_data["observation"]

    done = False
    steps = 0
    history: list[dict[str, Any]] = []

    while not done:
        if client is not None:
            action = llm_pick_action(
                client, model, task_description, observation, steps, max_epochs, history
            )
        else:
            action = sample_random_action()

        r = requests.post(f"{host}/step", json=action, timeout=60)
        r.raise_for_status()
        resp = r.json()
        observation = resp["observation"]
        done = bool(resp["done"])
        steps += 1

        history.append(observation)
        acc = observation.get("val_accuracy", 0)
        loss = observation.get("val_loss", 0)
        print(f"    Epoch {steps}: val_acc={acc:.4f} val_loss={loss:.4f}", flush=True)

    r = requests.post(f"{host}/grader", timeout=60)
    r.raise_for_status()
    return float(r.json()["score"]), steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OptimusEnv baseline agent")
    parser.add_argument("--host", default="http://localhost:7860")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    random.seed(args.seed)

    client = create_openai_client()
    agent_type = "LLM" if client else "Random"

    print(f"\n{'='*50}")
    print(f"OptimusEnv Baseline Agent ({agent_type})")
    print(f"{'='*50}")
    print(f"Host:  {args.host}")
    print(f"Seed:  {args.seed}")
    if client:
        print(f"Model: {args.model}")
    else:
        print("NOTE:  Set OPENAI_API_KEY to use LLM agent")
    print("-" * 50)

    scores: dict[str, float] = {}
    total_time = 0.0

    for task_id in ["task_1", "task_2", "task_3"]:
        label = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}[task_id]
        print(f"\n[{task_id}] ({label})", flush=True)
        t0 = time.time()
        score, steps = run_task(args.host, task_id, client, args.model)
        elapsed = time.time() - t0
        total_time += elapsed
        scores[task_id] = score
        print(f"  → score={score:.4f}  steps={steps}  time={elapsed:.1f}s")

    avg = sum(scores.values()) / 3
    print(f"\n{'='*50}")
    print(f"Results ({agent_type} Agent)")
    print(f"{'='*50}")
    print(f"Task 1 (easy):    {scores['task_1']:.4f}")
    print(f"Task 2 (medium):  {scores['task_2']:.4f}")
    print(f"Task 3 (hard):    {scores['task_3']:.4f}")
    print(f"Average:          {avg:.4f}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
