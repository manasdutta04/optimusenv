# OptimusEnv — PyTorch Training Optimization RL Environment

**An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment where AI agents learn to be better ML engineers.**

Every production ML team spends weeks tuning hyperparameters, selecting architectures, and optimizing training pipelines. This costs real money — [Google estimates](https://research.google/blog/) that hyperparameter tuning accounts for up to 40% of total training compute. OptimusEnv turns this real-world task into a structured RL problem: the agent receives live training metrics (loss, accuracy, throughput, memory) and must make intelligent decisions about learning rate, optimizer, architecture, and regularization to maximize model quality while minimizing resource usage.

Unlike toy environments, **every `step()` runs a real PyTorch training epoch** — `nn.Module.forward()`, `loss.backward()`, `optimizer.step()` — on real datasets (MNIST, FashionMNIST, CIFAR-10). The agent gets genuine, noisy training signals and must generalize its optimization strategy across tasks of increasing difficulty.

## Why This Matters

| Pain Point | How OptimusEnv Helps |
|---|---|
| ML engineers spend days on manual hyperparameter search | Trains RL agents to automate this process |
| Grid/random search wastes compute | RL agents learn informed search strategies from feedback |
| AutoML tools are black-box and expensive | OptimusEnv provides a transparent, reproducible training ground |
| No standardized benchmark for "training optimization" agents | OpenEnv-compliant API enables consistent evaluation |

## Environment Description

OptimusEnv exposes a stateful episode API over FastAPI. Each `step()` runs one real training epoch plus a full validation pass. The action controls optimizer selection, model depth/width, regularization, learning-rate schedule, and batch size. The environment returns observations (training metrics) and shaped rewards, then a task-specific grader computes a final normalized score in `[0.0, 1.0]`.

Key design properties:
- **Real training** — actual PyTorch forward/backward passes, not simulated numbers
- **CPU-safe** — runs on HF Spaces free tier (no GPU required)
- **Fast** — each `step()` completes in <1 second on CPU via dataset subsampling
- **Deterministic** — seeded randomness for reproducible baseline scores

## Observation Space

| Field | Type | Description |
|---|---|---|
| `epoch` | int | Current epoch number in the episode |
| `train_loss` | float | Mean training loss for the epoch |
| `val_loss` | float | Mean validation loss for the epoch |
| `val_accuracy` | float | Validation accuracy (0 to 1) |
| `throughput_sps` | float | Training throughput in samples per second |
| `mem_mb` | float | Peak memory usage in MB |
| `time_elapsed_sec` | float | Wall-clock duration of the epoch |
| `current_config` | dict | Effective action/config applied in this step |

## Action Space

| Field | Type | Range | Description |
|---|---|---|---|
| `learning_rate` | float | `1e-5` to `0.1` | Optimizer learning rate |
| `batch_size` | int | `8` to `512` | Batch size for train/val loaders |
| `weight_decay` | float | `0.0` to `0.1` | L2 regularization |
| `optimizer` | categorical | `adam`, `sgd`, `adamw` | Optimizer family |
| `num_layers` | int | `1` to `5` | Number of hidden blocks in head/MLP |
| `hidden_dim` | int | `32` to `512` | Hidden dimension for MLP/CNN head |
| `use_amp` | bool | `true`/`false` | Mixed precision (CUDA only) |
| `lr_schedule` | categorical | `none`, `cosine`, `step` | Learning-rate schedule |

## Tasks

### Task 1 — Easy: Hyperparameter Tuning on MNIST
> Fixed architecture (2 layers, 128 hidden). Agent only tunes learning rate, optimizer, batch size, regularization, and LR schedule.

- **Dataset:** MNIST (5,000 train / 1,000 val)
- **Target:** 92% validation accuracy
- **Grading:** `score = min(best_val_accuracy / 0.92, 1.0)`

### Task 2 — Medium: Generalization Across Datasets
> Full control over architecture AND training. Agent must find a model that works well on both MNIST and FashionMNIST simultaneously.

- **Datasets:** MNIST + FashionMNIST combined (10,000 train / 2,000 val)
- **Target:** 88% validation accuracy
- **Grading:** `score = min(final_accuracy / 0.88, 1.0) × epoch_penalty`
- **Penalty:** Score scaled by `epochs_run / 10` if fewer than 10 epochs completed

### Task 3 — Hard: Composite Optimization on CIFAR-10
> Maximize accuracy AND training efficiency. Uses a CNN architecture. Agent must balance quality vs resource usage.

- **Dataset:** CIFAR-10 (8,000 train / 2,000 val)
- **Target:** 60% validation accuracy
- **Grading:** `score = 0.5 × accuracy_score + 0.3 × speed_score + 0.2 × memory_score`
  - `accuracy_score = min(best_accuracy / 0.60, 1.0)`
  - `speed_score = min(best_throughput / 1000, 1.0)`
  - `memory_score = max(0, 1 - min_memory_mb / 1024)`

## Reward Function

Per-step shaped reward with partial progress signals:

```
acc_delta    = current_val_accuracy - previous_val_accuracy
acc_reward   = 2.0 × acc_delta
speed_bonus  = max(0, throughput_sps / 500 - 1) × 0.15
mem_bonus    = max(0, 1 - mem_mb / 512) × 0.10
reward       = clip(acc_reward + speed_bonus + mem_bonus, -1.0, 1.0)
```

- **Crash penalty:** If training crashes (NaN loss or OOM), reward = `-1.0` and the episode ends immediately
- **Partial progress:** Even small accuracy improvements generate positive reward
- **Efficiency bonuses:** Fast throughput and low memory usage provide additional reward signal

## Setup

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

### Docker

```bash
docker build -t optimusenv .
docker run --rm -p 7860:7860 optimusenv
```

### Running the Baseline Agent

```bash
# LLM agent (recommended — uses OpenAI API):
OPENAI_API_KEY=sk-... python baseline/run_baseline.py --host http://localhost:7860

# Random agent fallback (no API key needed):
python baseline/run_baseline.py --host http://localhost:7860
```

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health and version check |
| `POST` | `/reset` | Start a new episode for a task |
| `POST` | `/step` | Run one training epoch with given action |
| `GET` | `/state` | Inspect current environment state |
| `GET` | `/tasks` | List available tasks and action schema |
| `POST` | `/grader` | Score the completed episode (0.0–1.0) |
| `POST` | `/baseline` | Run internal random baseline across all 3 tasks |

## Example Usage

```bash
# 1. Health check
curl http://localhost:7860/health

# 2. List tasks
curl http://localhost:7860/tasks

# 3. Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"task_1"}'

# 4. Run a training step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"learning_rate":0.001,"batch_size":64,"optimizer":"adamw","num_layers":2,"hidden_dim":128,"use_amp":false,"lr_schedule":"none","weight_decay":0.0001}'

# 5. Repeat step until done=true, then grade
curl -X POST http://localhost:7860/grader
```

## Baseline Scores

| Agent | Task 1 | Task 2 | Task 3 | Average |
|---|---:|---:|---:|---:|
| Random (seed=42) | — | — | — | — |
| LLM (gpt-4o-mini) | — | — | — | — |

> Run `POST /baseline` or `python baseline/run_baseline.py` to generate scores on your hardware.

## HuggingFace Spaces Deployment

1. Create a new **Docker Space** on [HuggingFace](https://huggingface.co/new-space)
2. Push this repository (with `Dockerfile`, `openenv.yaml`, and all source code)
3. The container exposes port `7860` — HF Spaces handles routing automatically
4. After deploy, validate with:
   ```bash
   curl https://YOUR-SPACE.hf.space/health
   curl -X POST https://YOUR-SPACE.hf.space/reset -H "Content-Type: application/json" -d '{"task":"task_1"}'
   ```
5. Tag the Space with `openenv` for discoverability

## License

MIT
