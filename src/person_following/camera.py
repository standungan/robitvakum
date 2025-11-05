from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import CameraConfig

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - exercised only without HW support
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOG = logging.getLogger(__name__)


class RealSenseUnavailableError(RuntimeError):
    """Raised when pyrealsense2 is not available in the environment."""

@dataclass
class FrameBundle:
    """Container for synchronized color and depth frames."""

    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: "rs.intrinsics"  # type: ignore[name-defined]


class RealSenseCamera:
    """Minimal RealSense pipeline wrapper returning color and depth frames."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        if rs is None:
            message = (
                "pyrealsense2 is not installed. Install librealsense SDK and "
                "pyrealsense2 before running the follower pipeline."
            )
            raise RealSenseUnavailableError(message) from _IMPORT_ERROR

        self._config = config or CameraConfig()
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._depth_scale: float = 1.0
        self._profile: rs.pipeline_profile | None = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._pipeline is not None:
            return

        camera_cfg = self._config
        pipeline = rs.pipeline()
        config = rs.config()

        if camera_cfg.enable_color:
            config.enable_stream(
                rs.stream.color,
                camera_cfg.width,
                camera_cfg.height,
                rs.format.bgr8,
                camera_cfg.fps,
            )

        if camera_cfg.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                camera_cfg.width,
                camera_cfg.height,
                rs.format.z16,
                camera_cfg.fps,
            )

        profile = pipeline.start(config)
        LOG.info("RealSense pipeline started with %sx%s @ %s FPS", camera_cfg.width, camera_cfg.height, camera_cfg.fps)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        LOG.debug("Depth scale resolved to %.6f meters", self._depth_scale)

        self._align = rs.align(rs.stream.color)
        self._pipeline = pipeline
        self._profile = profile

    def stop(self) -> None:
        if self._pipeline is None:
            return

        LOG.info("Stopping RealSense pipeline")
        self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._profile = None

    def frames(self, *, timeout_ms: int = 5000) -> FrameBundle | None:
        if self._pipeline is None or self._align is None:
            raise RuntimeError("Camera pipeline is not running. Call start() first.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            LOG.warning("Incomplete frame set from RealSense pipeline")
            return None

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale

        intrinsics = (
            color_frame.profile.as_video_stream_profile().get_intrinsics()
        )
        return FrameBundle(color_bgr=color, depth_m=depth, intrinsics=intrinsics)

    @contextlib.contextmanager
    def streaming(self) -> Iterator["RealSenseCamera"]:
        try:
            self.start()
            yield self
        finally:
            self.stop()

    @property
    def profile(self) -> "rs.pipeline_profile":  # type: ignore[name-defined]
        if self._profile is None:
            raise RuntimeError("Pipeline profile unavailable before start().")
        return self._profile
    

