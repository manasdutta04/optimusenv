#!/usr/bin/env python3
import asyncio
import json
import os
import textwrap
import socket
import subprocess
import time
import traceback
from typing import Any, List, Optional

import requests
from openai import AsyncOpenAI

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

def wait_for_server(host="localhost", port=8000, timeout=60):
    """Block until the server is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                print(f"[INFO] Server at {host}:{port} is up.", flush=True)
                return True
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"Server at {host}:{port} did not start within {timeout}s")

async def get_model_action(client: AsyncOpenAI, task_description: str, observation: dict, history: list) -> dict:
    prompt = {
        "task_description": task_description,
        "observation": observation,
        "history_tail": history[-4:],
        "instruction": "Return ONLY the next action as valid JSON."
    }
    try:
        completion = await client.chat.completions.create(
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
        print(f"[ERROR] LLM Request failed: {exc}", flush=True)
        if hasattr(exc, "response"):
            print(f"[DEBUG] Response status: {exc.response.status_code}", flush=True)
            print(f"[DEBUG] Response details: {exc.response.text}", flush=True)
        raise exc

async def run_episode(client: AsyncOpenAI, task_id: str):
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
        action_dict = await get_model_action(client, env.task_description, observation, history)
        
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
    # --- START SERVER IN BACKGROUND ---
    # We use 0.0.0.0 to ensure it's reachable on all interfaces if needed
    host_addr = "0.0.0.0"
    port_num = "8000"
    server_proc = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", host_addr, "--port", port_num, "--workers", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[INFO] Started uvicorn server process on {host_addr}:{port_num}", flush=True)
    
    try:
        # Wait for the server to be ready locally
        wait_for_server(host="localhost", port=int(port_num), timeout=60)

        # Check environment connectivity to verify reset/baseline endpoints
        print(f"[DEBUG] Checking connectivity to environment server at {HOST}...", flush=True)
        try:
            requests.get(f"{HOST}/baseline", timeout=5)
            print("[DEBUG] Environment server reachable.", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not reach environment server: {e}", flush=True)

        async_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
        # Run tasks sequentially
        tasks = ["task_1", "task_2", "task_3"]
        all_scores = []
        
        for task_id in tasks:
            score = await run_episode(async_client, task_id)
            all_scores.append(score)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"\nFinal Average Score: {avg_score:.4f}")

    except Exception:
        print("\n" + "="*50)
        print("CRITICAL ERROR: Unhandled exception in main()")
        print("="*50)
        traceback.print_exc()
        print("="*50 + "\n")
        # Exit with error to notify validator
        os._exit(1)
    finally:
        print("[INFO] Terminating server process.", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
