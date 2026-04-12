#!/usr/bin/env python3
import asyncio
import json
import os
import textwrap
import time
from typing import Any, List, Optional

import requests
from openai import OpenAI

# Environment variables as per mandatory requirements
# Using direct os.environ access to force failure if mandatory variables are missing
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    # Ensure /v1 suffix for OpenAI client compatibility with proxies
    # This is critical as many proxies listen on /v1/chat/completions
    if not API_BASE_URL.endswith("/v1") and not API_BASE_URL.endswith("/v1/"):
        API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"
except KeyError:
    print("[WARNING] API_BASE_URL missing in environment, falling back to router.", flush=True)
    API_BASE_URL = "https://router.huggingface.co/v1"

try:
    API_KEY = os.environ["API_KEY"]
except KeyError:
    print("[WARNING] API_KEY missing in environment, falling back to HF_TOKEN.", flush=True)
    API_KEY = os.environ.get("HF_TOKEN", "")

MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
HOST = os.getenv("HOST", "http://localhost:8000")

# Diagnostic logging to participant logs (masked)
print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", flush=True)
masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if API_KEY and len(API_KEY) > 8 else "****"
print(f"[DEBUG] Using API_KEY: {masked_key}", flush=True)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert ML engineer optimizing a real PyTorch training run.
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
    Prefer small, sensible adjustments over drastic jumps.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

class OptimusEnvWrapper:
    def __init__(self, host: str, task_id: str):
        self.host = host
        self.task_id = task_id
        self.max_epochs = 0
        self.task_description = ""

    async def reset(self):
        resp = requests.post(f"{self.host}/reset", json={"task": self.task_id}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        self.max_epochs = data["max_epochs"]
        self.task_description = data["task_description"]
        return data

    async def step(self, action: dict):
        resp = requests.post(f"{self.host}/step", json=action, timeout=120)
        resp.raise_for_status()
        return resp.json()

    async def get_score(self) -> float:
        resp = requests.post(f"{self.host}/grader", timeout=60)
        resp.raise_for_status()
        return float(resp.json()["score"])

    async def close(self):
        # Optional cleanup
        pass

def get_model_action(client: OpenAI, task_description: str, observation: dict, history: list) -> dict:
    prompt = {
        "task_description": task_description,
        "observation": observation,
        "history_tail": history[-4:],
        "instruction": "Return ONLY the next action as valid JSON."
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        content = (completion.choices[0].message.content or "").strip()
        # Clean markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
        return json.loads(content)
    except Exception as exc:
        # Logging failure to stdout and RAISING it to force a Phase 2 execution failure
        # This prevents silent heuristic fallbacks and reveals the true error in logs.
        print(f"[ERROR] LLM Request failed: {exc}", flush=True)
        if hasattr(exc, "response"):
            print(f"[DEBUG] Response status: {exc.response.status_code}", flush=True)
            print(f"[DEBUG] Response details: {exc.response.text}", flush=True)
        raise exc

async def run_episode(client: OpenAI, task_id: str):
    env = OptimusEnvWrapper(HOST, task_id)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="optimusenv", model=MODEL_NAME)

    reset_data = await env.reset()
    observation = reset_data["observation"]
    max_epochs = reset_data["max_epochs"]

    for step in range(1, max_epochs + 1):
        action_dict = get_model_action(client, env.task_description, observation, history)
        
        step_data = await env.step(action_dict)
        observation = step_data["observation"]
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        
        rewards.append(reward)
        steps_taken = step
        
        action_str = json.dumps(action_dict)
        log_step(step=step, action=action_str, reward=reward, done=done, error=None)
        
        history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")
        if done:
            break
    
    score = await env.get_score()
    success = score >= 0.1  # Success threshold as per user example
    
    await env.close()
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run tasks sequentially
    tasks = ["task_1", "task_2", "task_3"]
    all_scores = []
    
    for task_id in tasks:
        score = await run_episode(client, task_id)
        all_scores.append(score)
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\nFinal Average Score: {avg_score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
