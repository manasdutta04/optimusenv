from __future__ import annotations
import os
import asyncio
from typing import Any, Optional

# We try to import from the official core package
try:
    from openenv.core import HTTPEnvClient
except ImportError:
    # Fallback to the legacy naming if needed, though 3.13 should have the new one
    try:
        from openenv_core import HTTPEnvClient
    except ImportError:
        # If all else fails, we define a minimal compatible base
        class HTTPEnvClient:
            def __init__(self, base_url: str):
                self.base_url = base_url
            @classmethod
            async def from_docker_image(cls, image_name: str, port: int = 8000):
                # This is a stub for local testing if the SDK is missing
                print(f"[SDK STUB] Starting container from {image_name}...")
                return cls(base_url=f"http://localhost:{port}")

class OptimusEnvEnv(HTTPEnvClient):
    """SDK client for OptimusEnv."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(base_url)

    # Note: from_docker_image is usually provided by the base class in the SDK
    # but we ensure it's available and compatible with the user template.
    
    async def reset(self, task: str = "task_1"):
        # The official SDK usually does this via POST /reset
        # In the template, it's called as 'await env.reset()'
        import requests
        resp = requests.post(f"{self.base_url}/reset", json={"task": task}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Mocking the attribute access the template expects
        from dataclasses import dataclass
        @dataclass
        class ResetResult:
            observation: Any
            task_description: str
            max_epochs: int
        
        return ResetResult(
            observation=data["observation"],
            task_description=data["task_description"],
            max_epochs=data["max_epochs"]
        )

    async def step(self, action: Any):
        import requests
        # Convert Pydantic model to dict if needed
        action_data = action.model_dump() if hasattr(action, "model_dump") else action
        resp = requests.post(f"{self.base_url}/step", json=action_data, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        from dataclasses import dataclass
        @dataclass
        class StepResult:
            observation: Any
            reward: float
            done: bool
            info: dict
        
        return StepResult(
            observation=data["observation"],
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {})
        )

    async def grader(self):
        import requests
        resp = requests.post(f"{self.base_url}/grader", timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        from dataclasses import dataclass
        @dataclass
        class GraderResult:
            score: float
        
        return GraderResult(score=data["score"])

    async def close(self):
        # The base class might handle container shutdown
        if hasattr(super(), "close"):
            await super().close()
