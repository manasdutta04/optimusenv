#!/usr/bin/env python3
"""
inference.py — OptimusEnv submission for Scaler x Meta PyTorch OpenEnv Hackathon
Using the official OpenEnv SDK pattern with from_docker_image().
"""

import asyncio
import json
import os
import textwrap
import traceback
from typing import List, Optional

from openai import AsyncOpenAI
from optimusenv import OptimusEnvAction, OptimusEnvEnv

# Environment variables
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Ensure /v1 for LiteLLM proxy compatibility
if API_BASE_URL and not API_BASE_URL.endswith("/v1") and not API_BASE_URL.endswith("/v1/"):
    API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", flush=True)
print(f"[DEBUG] Using IMAGE_NAME: {IMAGE_NAME}", flush=True)

SYSTEM_PROMPT = textwrap.dedent("""
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
    Return ONLY the JSON object, no markdown, no explanation.
""").strip()

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

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
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
        return json.loads(content)
    except Exception as exc:
        print(f"[WARN] LLM failed ({exc}), using default action", flush=True)
        return {
            "learning_rate": 1e-3, "batch_size": 64, "weight_decay": 1e-4,
            "optimizer": "adamw", "num_layers": 2, "hidden_dim": 128,
            "use_amp": False, "lr_schedule": "cosine"
        }

async def run_episode(client: AsyncOpenAI, env: OptimusEnvEnv, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history = []

    log_start(task=task_id, env="optimusenv", model=MODEL_NAME)

    try:
        reset_result = await env.reset(task=task_id)
        # Handle both object attributes and dict-like access
        observation = reset_result.observation if hasattr(reset_result, "observation") else reset_result["observation"]
        task_description = getattr(reset_result, "task_description", task_id)
        max_epochs = getattr(reset_result, "max_epochs", 5)

        for step in range(1, max_epochs + 1):
            action_dict = await get_model_action(client, task_description, observation, history)
            action = OptimusEnvAction(**action_dict)

            result = await env.step(action)
            observation = result.observation if hasattr(result, "observation") else result["observation"]
            reward = float(result.reward if hasattr(result, "reward") else result.get("reward", 0.0))
            done = result.done if hasattr(result, "done") else result.get("done", False)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=json.dumps(action_dict), reward=reward, done=done, error=None)
            history.append(f"Step {step}: {json.dumps(action_dict)} -> reward {reward:.2f}")

            if done:
                break

        try:
            grader_result = await env.grader()
            score = float(grader_result.score if hasattr(grader_result, "score") else grader_result.get("score", 0.0))
        except Exception:
            score = sum(rewards) / len(rewards) if rewards else 0.0

        success = score >= 0.1

    except Exception as e:
        print(f"[ERROR] Episode {task_id} failed!", flush=True)
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

async def main():
    async_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Initializing environment from image: {IMAGE_NAME}", flush=True)
    try:
        env = await OptimusEnvEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(f"[CRITICAL] Failed to start environment from image: {e}", flush=True)
        traceback.print_exc()
        return

    all_scores = []
    try:
        for task_id in ["task_1", "task_2", "task_3"]:
            score = await run_episode(async_client, env, task_id)
            all_scores.append(score)
    except Exception as e:
        print(f"[CRITICAL] Error during tasks: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    if all_scores:
        avg = sum(all_scores)/len(all_scores)
        print(f"\nFinal Average Score: {avg:.4f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
