"""OpenEnv-facing model aliases for the OptimusEnv package."""

from app.models import Action as OptimusAction
from app.models import Observation as OptimusObservation

__all__ = [
    "OptimusAction",
    "OptimusObservation",
]
