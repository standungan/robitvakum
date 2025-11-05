"""
Person following utilities built around Intel RealSense depth cameras.
"""

from .camera import RealSenseCamera, RealSenseUnavailableError
from .config import (
    CameraConfig,
    ControllerConfig,
    DetectorConfig,
    FollowerConfig,
)
from .controller import SimpleFollowerController
from .detector import PersonDetector
from .tracker import TargetEstimator, TargetEstimate

__all__ = [
    "RealSenseCamera",
    "RealSenseUnavailableError",
    "CameraConfig",
    "DetectorConfig",
    "ControllerConfig",
    "FollowerConfig",
    "PersonDetector",
    "SimpleFollowerController",
    "TargetEstimator",
    "TargetEstimate",
]
