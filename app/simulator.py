from __future__ import annotations

import os
import time
import tracemalloc
from contextlib import nullcontext
from typing import Any, Optional

import certifi
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from app.models import Action


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(max(0, num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SmallCIFARClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        head_layers: list[nn.Module] = []
        in_dim = 64
        for _ in range(max(1, num_layers)):
            head_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 10))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.head(x)


class TrainingSimulator:
    def __init__(self, task_config: dict[str, Any]) -> None:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        self.task = task_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = os.environ.get("TORCHVISION_DATASETS", "./data")
        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]
        self.dataset_kind: str = "mnist"
        self._prepare_data()

    @staticmethod
    def _subset_dataset(dataset: Dataset[Any], size: int, seed: int = 42) -> Subset[Any]:
        generator = torch.Generator().manual_seed(seed)
        n = len(dataset)
        subset_size = min(size, n)
        indices = torch.randperm(n, generator=generator)[:subset_size].tolist()
        return Subset(dataset, indices)

    def _load_dataset(self, name: str) -> tuple[Dataset[Any], Dataset[Any], str]:
        if name == "mnist":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            train = datasets.MNIST(
                root=self.data_root,
                train=True,
                transform=transform,
                download=True,
            )
            val = datasets.MNIST(
                root=self.data_root,
                train=False,
                transform=transform,
                download=True,
            )
            return (
                self._subset_dataset(train, 5000),
                self._subset_dataset(val, 1000),
                "mnist",
            )

        if name == "fashion_mnist":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)),
                ]
            )
            train = datasets.FashionMNIST(
                root=self.data_root,
                train=True,
                transform=transform,
                download=True,
            )
            val = datasets.FashionMNIST(
                root=self.data_root,
                train=False,
                transform=transform,
                download=True,
            )
            return (
                self._subset_dataset(train, 5000),
                self._subset_dataset(val, 1000),
                "fashion_mnist",
            )

        if name == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616),
                    ),
                ]
            )
            train = datasets.CIFAR10(
                root=self.data_root,
                train=True,
                transform=transform,
                download=True,
            )
            val = datasets.CIFAR10(
                root=self.data_root,
                train=False,
                transform=transform,
                download=True,
            )
            return (
                self._subset_dataset(train, 8000),
                self._subset_dataset(val, 2000),
                "cifar10",
            )

        raise ValueError(f"Unsupported dataset: {name}")

    def _prepare_data(self) -> None:
        if "datasets" in self.task:
            train_parts: list[Dataset[Any]] = []
            val_parts: list[Dataset[Any]] = []
            for ds_name in self.task["datasets"]:
                train_ds, val_ds, _ = self._load_dataset(ds_name)
                train_parts.append(train_ds)
                val_parts.append(val_ds)
            self.train_dataset = ConcatDataset(train_parts)
            self.val_dataset = ConcatDataset(val_parts)
            self.dataset_kind = "mnist"
            return

        dataset_name = self.task.get("dataset", "mnist")
        self.train_dataset, self.val_dataset, self.dataset_kind = self._load_dataset(
            dataset_name
        )

    def create_dataloaders(self, batch_size: int) -> tuple[DataLoader[Any], DataLoader[Any]]:
        safe_batch_size = max(1, min(batch_size, len(self.train_dataset)))
        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=safe_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=safe_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    def build_model(self, action: Action) -> nn.Module:
        if self.dataset_kind == "cifar10":
            model = SmallCIFARClassifier(
                hidden_dim=action.hidden_dim,
                num_layers=action.num_layers,
            )
        else:
            model = MLPClassifier(
                hidden_dim=action.hidden_dim,
                num_layers=action.num_layers,
            )
        return model.to(self.device)

    def run_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        action: Action,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> dict[str, Any]:
        use_amp = bool(action.use_amp and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        criterion = nn.CrossEntropyLoss()

        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                tracemalloc.start()

            wall_start = time.perf_counter()
            model.train()

            train_loss_total = 0.0
            train_samples = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", enabled=True)
                    if use_amp
                    else nullcontext()
                )
                with amp_ctx:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if not torch.isfinite(loss):
                    return {"crashed": True, "reason": "nan_loss"}

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_n = targets.size(0)
                train_samples += batch_n
                train_loss_total += float(loss.item()) * batch_n

            train_loss = train_loss_total / max(1, train_samples)

            model.eval()
            val_loss_total = 0.0
            val_samples = 0
            correct = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    amp_ctx = (
                        torch.amp.autocast(device_type="cuda", enabled=True)
                        if use_amp
                        else nullcontext()
                    )
                    with amp_ctx:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    if not torch.isfinite(loss):
                        return {"crashed": True, "reason": "nan_loss"}

                    batch_n = targets.size(0)
                    val_samples += batch_n
                    val_loss_total += float(loss.item()) * batch_n
                    preds = outputs.argmax(dim=1)
                    correct += int((preds == targets).sum().item())

            if scheduler is not None:
                scheduler.step()

            elapsed = time.perf_counter() - wall_start

            if self.device.type == "cuda":
                mem_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
            else:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_mb = float(peak / (1024 ** 2))

            val_loss = val_loss_total / max(1, val_samples)
            val_accuracy = correct / max(1, val_samples)
            throughput_sps = train_samples / max(elapsed, 1e-8)

            return {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "throughput_sps": float(throughput_sps),
                "mem_mb": float(mem_mb),
                "time_elapsed_sec": float(elapsed),
                "crashed": False,
            }
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                return {"crashed": True, "reason": "oom"}
            raise
        finally:
            if self.device.type != "cuda" and tracemalloc.is_tracing():
                tracemalloc.stop()

    def build_optimizer(self, model: nn.Module, action: Action) -> optim.Optimizer:
        params = model.parameters()
        if action.optimizer == "adam":
            return optim.Adam(
                params,
                lr=action.learning_rate,
                weight_decay=action.weight_decay,
            )
        if action.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=action.learning_rate,
                momentum=0.9,
                weight_decay=action.weight_decay,
            )
        return optim.AdamW(
            params,
            lr=action.learning_rate,
            weight_decay=action.weight_decay,
        )

    def build_scheduler(
        self,
        optimizer: optim.Optimizer,
        action: Action,
        max_epochs: int,
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        if action.lr_schedule == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max_epochs)
        if action.lr_schedule == "step":
            return StepLR(optimizer, step_size=max(1, max_epochs // 3), gamma=0.5)
        return None
