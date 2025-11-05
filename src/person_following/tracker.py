from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .detector import Detection

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - exercised only without HW support
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@dataclass
class TargetEstimate:
    """3D estimate of the tracked person in camera coordinates."""
    detection: Detection
    distance_m: float
    bearing_rad: float
    elevation_rad: float
    position_m: tuple[float, float, float]
    valid_depth_ratio: float

    @property
    def bearing_deg(self) -> float:
        return math.degrees(self.bearing_rad)

    @property
    def elevation_deg(self) -> float:
        return math.degrees(self.elevation_rad)


class TargetEstimator:
    """Converts 2D detections into 3D camera-relative coordinates."""

    def __init__(self, depth_kernel: int = 5, min_valid_ratio: float = 0.3) -> None:
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required to run TargetEstimator. Install librealsense first."
            ) from _IMPORT_ERROR
        if depth_kernel % 2 == 0:
            raise ValueError("depth_kernel must be an odd number.")
        self._depth_kernel = depth_kernel
        self._min_valid_ratio = min_valid_ratio

    def estimate(
        self,
        detection: Detection,
        depth_m: np.ndarray,
        intrinsics: "rs.intrinsics",  # type: ignore[name-defined]
    ) -> Optional[TargetEstimate]:
        """Return the 3D position for a single detection or None if not enough depth."""
        height, width = depth_m.shape
        cx, cy = detection.center

        half = self._depth_kernel // 2
        x_start = max(0, cx - half)
        x_end = min(width, cx + half + 1)
        y_start = max(0, cy - half)
        y_end = min(height, cy + half + 1)

        depth_window = depth_m[y_start:y_end, x_start:x_end]
        if depth_window.size == 0:
            return None

        valid_mask = depth_window > 0.05  # ignore near-zero depth noise
        valid_depths = depth_window[valid_mask]
        valid_ratio = float(valid_depths.size) / float(depth_window.size)
        if valid_ratio < self._min_valid_ratio:
            return None

        distance = float(np.median(valid_depths))
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], distance)
        x, y, z = point

        if distance <= 0.0 or z <= 0.0:
            return None

        bearing = math.atan2(x, z)
        elevation = math.atan2(-y, z)  # RealSense frame has Y pointing down

        return TargetEstimate(
            detection=detection,
            distance_m=distance,
            bearing_rad=float(bearing),
            elevation_rad=float(elevation),
            position_m=(float(x), float(y), float(z)),
            valid_depth_ratio=valid_ratio,
        )

