# OptimusEnv — Agent Reference

> **OpenEnv-compliant RL environment for PyTorch training optimization.**  
> Hackathon submission: Scaler × Meta PyTorch OpenEnv Hackathon.

## Architecture

OptimusEnv is a **stateful REST API server** built with FastAPI. An RL agent interacts with it by:
1. **Resetting** (`POST /reset`) — chooses one of 3 tasks, initializes a training session
2. **Stepping** (`POST /step`) — sends hyperparameter/architecture decisions as actions; server runs one **real PyTorch training epoch** and returns observations
3. **Grading** (`POST /grader`) — after episode completion, returns a normalized score `[0.0, 1.0]`

All training is **real** — `nn.Module.forward()`, `loss.backward()`, and `optimizer.step()` execute every call. No mocked numbers.

## File Map

```
optimusenv/
├── app/
│   ├── __init__.py         # package init
│   ├── models.py           # Pydantic v2 models: Action, Observation, StepResponse, etc.
│   ├── tasks.py            # 3 task configs (easy/medium/hard) as dict constants
│   ├── simulator.py        # Real PyTorch training harness — MLPClassifier, SmallCIFARClassifier
│   ├── environment.py      # Stateful env class: reset(), step(), compute_reward(), get_state()
│   ├── graders.py          # 3 grader functions + get_grader() dispatcher
│   └── main.py             # FastAPI app, 7 endpoints, startup dataset caching
├── baseline/
│   └── run_baseline.py     # Standalone random agent (HTTP client)
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # HF Spaces deployment
├── requirements.txt        # Python deps (torch, fastapi, openenv-core, etc.)
├── README.md               # Full documentation (12 sections)
└── agent.md                # This file
```

## Key Components

### `simulator.py` — Core Training Engine
- **`TrainingSimulator`** — loads torchvision datasets (MNIST/FashionMNIST/CIFAR-10) with small subsets for speed
- **`MLPClassifier`** — Flatten → Linear → BN → ReLU layers for 28×28 images
- **`SmallCIFARClassifier`** — Conv2d backbone + linear head for 32×32 images
- `run_epoch()` — trains one epoch + validates; tracks wall-clock time, memory (tracemalloc on CPU)
- Handles NaN loss and OOM crashes gracefully

### `environment.py` — State Machine
- **`OptimusEnv`** — singleton managing one episode at a time
- On `step()`: detects if architecture/optimizer/batch_size changed → rebuilds only what's needed
- Computes per-step reward: `clip(2*acc_delta + speed_bonus + mem_bonus, -1, 1)`
- Locked params enforced per task (e.g., Task 1 locks `num_layers` and `hidden_dim`)

### `graders.py` — Scoring
| Grader | Task | Formula |
|---|---|---|
| `accuracy_grader` | Task 1 | `min(best_acc / 0.92, 1.0)` |
| `generalization_grader` | Task 2 | `min(final_acc / 0.88, 1.0) * epoch_penalty` |
| `composite_grader` | Task 3 | `0.5*acc + 0.3*speed + 0.2*memory` |

## API Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | `HealthResponse` |
| `POST` | `/reset` | `{"task": "task_1"}` | `ResetResponse` |
| `POST` | `/step` | `Action` JSON | `StepResponse` |
| `GET` | `/state` | — | `StateResponse` |
| `GET` | `/tasks` | — | `list[TaskInfo]` |
| `POST` | `/grader` | — | `GraderResponse` |
| `POST` | `/baseline` | — | `BaselineResponse` |

## Action Space

| Field | Type | Range | Default |
|---|---|---|---|
| `learning_rate` | float | `1e-5` — `0.1` | `0.001` |
| `batch_size` | int | `8` — `512` | `64` |
| `weight_decay` | float | `0.0` — `0.1` | `1e-4` |
| `optimizer` | enum | `adam`, `sgd`, `adamw` | `adamw` |
| `num_layers` | int | `1` — `5` | `2` |
| `hidden_dim` | int | `32` — `512` | `128` |
| `use_amp` | bool | — | `false` |
| `lr_schedule` | enum | `none`, `cosine`, `step` | `none` |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `epoch` | int | Current epoch (1-indexed) |
| `train_loss` | float | Mean training loss |
| `val_loss` | float | Mean validation loss |
| `val_accuracy` | float | Validation accuracy `[0, 1]` |
| `throughput_sps` | float | Samples/second |
| `mem_mb` | float | Peak memory in MB |
| `time_elapsed_sec` | float | Epoch wall-clock time |
| `current_config` | dict | Effective action config |

## Tasks

| ID | Difficulty | Dataset | Epochs | Target Acc | Grader |
|---|---|---|---|---|---|
| `task_1` | Easy | MNIST (5K train) | 10 | 0.92 | `accuracy_grader` |
| `task_2` | Medium | MNIST + FashionMNIST (10K train) | 15 | 0.88 | `generalization_grader` |
| `task_3` | Hard | CIFAR-10 (8K train) | 20 | 0.60 | `composite_grader` |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server locally
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1

# Run baseline agent
python baseline/run_baseline.py --host http://localhost:7860

# Docker build + run
docker build -t optimusenv .
docker run --rm -p 7860:7860 optimusenv
```

## Critical Design Decisions

1. **CPU-safe** — `torch.device("cuda" if ... else "cpu")`. Never calls `.cuda()` unconditionally.
2. **Fast epochs** — small dataset subsets ensure `step()` completes under 5s on CPU.
3. **Deterministic** — `torch.manual_seed(42)` and `random.seed(42)` at startup.
4. **Clean reset** — calling `/reset` twice resets all state including model weights.
5. **Crash-safe** — all endpoints wrapped in try/except; NaN/OOM detected and reported.
6. **Dataset caching** — all datasets pre-downloaded at startup via `@app.on_event("startup")`.
7. **SSL handling** — `certifi` used for reliable HTTPS dataset downloads.
