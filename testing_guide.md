# OptimusEnv Testing and Verification Guide

This guide provides step-by-step instructions for testing, verifying, and interacting with the **OptimusEnv** reinforcement learning environment.

## 📱 Live Environment
The official OpenEnv-compliant endpoint is hosted at:
**`https://manasdutta04-optimusenv.hf.space/`**

---

## 1. Quick Health Check (CURL)
To verify the environment is online and responsive:

```bash
curl https://manasdutta04-optimusenv.hf.space/health
```
**Expected Output:** `{"status":"ok", "env":"OptimusEnv", "version":"1.0.0"}`

---

## 2. Inspecting the Task Suite
List all available RL tasks, including their descriptions and required action schemas:

```bash
curl https://manasdutta04-optimusenv.hf.space/tasks
```

---

## 3. Running an Episode (Step-by-Step)
You can manually interact with the environment to verify the transition logic:

### Step 1: Start/Reset an Episode
Choose a task (e.g., `task_1`):
```bash
curl -X POST https://manasdutta04-optimusenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"task_1"}'
```

### Step 2: Perform an Action
Run one training epoch with specific hyperparameters:
```bash
curl -X POST https://manasdutta04-optimusenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "adamw",
    "num_layers": 2,
    "hidden_dim": 128,
    "use_amp": false,
    "lr_schedule": "cosine",
    "weight_decay": 0.0001
  }'
```

---

## 4. Comprehensive Testing Scripts
For full execution verification, use the provided Python scripts in the repository.

### A. Run the Heuristic Baseline
This runs a pre-defined policy across all three tasks and reports a final score:
```bash
# Point the script to the live URL
python baseline/run_baseline.py --host https://manasdutta04-optimusenv.hf.space
```

### B. Run the Submission Inference
Simulate an agent run (this uses the same interface as the hackathon validator):
```bash
export API_BASE_URL=https://manasdutta04-optimusenv.hf.space
python inference.py
```

---

## 💡 Troubleshooting and Tips

> [!TIP]
> **Dataset Pre-caching:** The first time a task is run, the environment may take a few seconds to download the relevant PyTorch datasets (MNIST, CIFAR-10, etc.). Subsequent steps/resets will be much faster.

> [!IMPORTANT]
> **Port Consistency:** If running locally, the environment serves on port **8000**. On Hugging Face Spaces, use the root URL provided.

> [!WARNING]
> **Crash Penalty:** Running an action with extremely high learning rates or too many layers (causing OOM) will result in a reward of `-1.0` and terminate the episode immediately.

## 📊 Interpreting Results
- **`/grader`**: This endpoint returns the final normalized score `[0.0 - 1.0]` for the last completed episode.
- **`/baseline`**: This runs a server-side heuristic test to provide a reference score for all tasks.

---
*For more technical details on the reward function and observation space, refer to the [README.md](./README.md).*
