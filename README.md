# OptimusEnv — PyTorch Training Optimization RL Environment

OptimusEnv is an OpenEnv-compliant REST environment where an RL agent acts like an ML engineer: it selects hyperparameters and architecture settings, triggers real PyTorch training, and receives live metrics (loss, accuracy, speed, memory) as observations. The motivation is practical: production training always involves accuracy-vs-cost tradeoffs, so this environment teaches policies that optimize both model quality and resource efficiency under realistic constraints.

## Environment Description

OptimusEnv exposes a stateful episode API over FastAPI. Each `step()` runs one real training epoch plus validation using PyTorch and torchvision datasets (MNIST, FashionMNIST, CIFAR-10 subsets). The action controls optimizer choices, model depth/width, regularization, learning-rate schedule, and batch size. The environment returns observations and rewards, then a task-specific grader computes a final normalized score in `[0.0, 1.0]`.

## Observation Space

| field | type | description |
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

| field | type | range | description |
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

- Task 1 (Easy): Tune hyperparameters on MNIST with fixed architecture (`num_layers=2`, `hidden_dim=128`). Grading uses best validation accuracy normalized to target `0.92`.
- Task 2 (Medium): Optimize architecture + training for generalization across combined MNIST and FashionMNIST. Grading uses final accuracy normalized to target `0.88`, with a short-episode penalty for fewer than 10 epochs.
- Task 3 (Hard): Optimize CIFAR-10 for both quality and efficiency. Grading is a weighted composite of accuracy (`50%`), throughput (`30%`), and memory efficiency (`20%`) with target accuracy `0.60`.

## Reward Function

Per-step reward:

```text
acc_delta = curr_val_accuracy - prev_val_accuracy
acc_reward = 2.0 * acc_delta
speed_bonus = max(0, throughput_sps / 500 - 1) * 0.15
mem_bonus = max(0, 1 - mem_mb / 512) * 0.10
reward = clip(acc_reward + speed_bonus + mem_bonus, -1.0, 1.0)
```

If training crashes (NaN/OOM), reward is `-1.0` and the episode ends.

## Setup

Local run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

Docker run:

```bash
docker build -t optimusenv .
docker run --rm -p 7860:7860 optimusenv
```

## API Reference

| method | path | description |
|---|---|---|
| `GET` | `/health` | Service health/version check |
| `POST` | `/reset` | Start a new episode for a task |
| `POST` | `/step` | Run one training epoch from an action |
| `GET` | `/state` | Inspect current environment state |
| `GET` | `/tasks` | List tasks and action schema |
| `POST` | `/grader` | Score the completed episode |
| `POST` | `/baseline` | Run internal random baseline across all tasks |

## Example Usage

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"task_1"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"learning_rate":0.001,"batch_size":64,"optimizer":"adamw","num_layers":2,"hidden_dim":128,"use_amp":false,"lr_schedule":"none","weight_decay":0.0001}'

# Repeat /step until done=true, then:
curl -X POST http://localhost:7860/grader
```

## Baseline Scores

| run | task_1 | task_2 | task_3 | average |
|---|---:|---:|---:|---:|
| random (seed=42) | — | — | — | — |

> Run `POST /baseline` on a live server to generate real scores. Scores depend on hardware and random seed.
> You can also run the standalone baseline agent: `python baseline/run_baseline.py --host http://localhost:7860`

## HuggingFace Spaces Deployment

1. Create a new Docker Space on HuggingFace.
2. Push this repository with `Dockerfile`, `openenv.yaml`, and application code.
3. Ensure container port is `7860`.
4. Space startup command is provided by Docker `CMD` (`uvicorn app.main:app ...`).
5. After deploy, validate `/health`, `/tasks`, `/reset`, `/step`, and `/baseline` from the Space URL.
