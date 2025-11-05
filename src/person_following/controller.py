from __future__ import annotations

from dataclasses import dataclass

from .config import ControllerConfig
from .tracker import TargetEstimate


@dataclass
class ControlCommand:
    linear_velocity_mps: float
    angular_velocity_radps: float
    distance_error_m: float
    angle_error_rad: float


class SimpleFollowerController:
    """P-controller that keeps the person centered and at a target distance."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self._config = config or ControllerConfig()

    def compute(self, estimate: TargetEstimate | None) -> ControlCommand:
        if estimate is None:
            return ControlCommand(
                linear_velocity_mps=0.0,
                angular_velocity_radps=0.0,
                distance_error_m=0.0,
                angle_error_rad=0.0,
            )

        cfg = self._config
        distance_error = estimate.distance_m - cfg.target_distance_m
        angle_error = estimate.bearing_rad

        linear = _clamp(distance_error * cfg.distance_gain, -cfg.max_linear_speed_mps, cfg.max_linear_speed_mps)
        angular = _clamp(angle_error * cfg.angle_gain, -cfg.max_angular_speed_radps, cfg.max_angular_speed_radps)

        return ControlCommand(
            linear_velocity_mps=linear,
            angular_velocity_radps=angular,
            distance_error_m=distance_error,
            angle_error_rad=angle_error,
        )


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)

