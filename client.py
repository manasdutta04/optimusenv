"""Minimal HTTP client for interacting with a running OptimusEnv server."""

from __future__ import annotations

from typing import Any

import requests

from .models import OptimusAction


class OptimusEnvClient:
    """Thin convenience client for the OptimusEnv HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def reset(self, task: str = "task_1") -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: OptimusAction) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def tasks(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def grader(self) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/grader", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def baseline(self) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/baseline", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
