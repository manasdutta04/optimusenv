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
# Prioritize API_KEY and API_BASE_URL over fallbacks to ensure proxy compliance
API_BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HOST = os.environ.get("HOST", "http://localhost:8000")

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
        # Logging failure to stdout to help diagnose proxy/connectivity issues
        print(f"[ERROR] LLM Request failed: {exc}", flush=True)
        if hasattr(exc, "response"):
            print(f"[DEBUG] Response status: {exc.response.status_code}", flush=True)
            print(f"[DEBUG] Response text: {exc.response.text}", flush=True)
        # Fallback to a safe default action
        return {
            "learning_rate": 0.001,
            "batch_size": 64,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "num_layers": 2,
            "hidden_dim": 128,
            "use_amp": False,
            "lr_schedule": "none"
        }

async def run_episode(client: OpenAI, task_id: str):
    env = OptimusEnvWrapper(HOST, task_id)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="optimusenv", model=MODEL_NAME)

    try:
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
    except Exception as e:
        print(f"[ERROR] Episode failed: {e}", flush=True)
    finally:
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
