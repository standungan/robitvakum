from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from .camera import FrameBundle, RealSenseCamera, RealSenseUnavailableError
from .config import FollowerConfig
from .controller import ControlCommand, SimpleFollowerController
from .detector import Detection, PersonDetector
from .tracker import TargetEstimate, TargetEstimator
from .visualization import annotate_frame, overlay_command

LOG = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Latest perception + control snapshot produced by the pipeline."""

    timestamp: float
    detection: Detection | None
    estimate: TargetEstimate | None
    command: ControlCommand
    annotated_frame_jpeg: bytes | None
    fps: float

class PersonFollowerPipeline:
    """Runs the person follower pipeline in a background thread."""

    def __init__(
        self,
        config: FollowerConfig,
        *,
        depth_kernel: int = 5,
        min_valid_ratio: float = 0.3,
        annotate: bool = True,
    ) -> None:
        self._config = config
        self._camera = RealSenseCamera(config.camera)
        self._detector = PersonDetector(config.detector)
        self._estimator = TargetEstimator(depth_kernel=depth_kernel, min_valid_ratio=min_valid_ratio)
        self._controller = SimpleFollowerController(config.controller)

        self._annotate = annotate
        self._state: PipelineState | None = None
        self._error: str | None = None
        self._fps: float = 0.0
        self._last_timestamp: float | None = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="PersonFollowerPipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest_state(self) -> PipelineState | None:
        with self._lock:
            return self._state

    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def _run_loop(self) -> None:
        LOG.info("Starting person follower pipeline thread")
        try:
            with self._camera.streaming():
                while not self._stop_event.is_set():
                    frame_bundle = self._camera.frames()
                    if frame_bundle is None:
                        continue
                    state = self._process(frame_bundle)
                    with self._lock:
                        self._state = state
                    time.sleep(0)  # yield to event loop
        except RealSenseUnavailableError as exc:
            LOG.error("RealSense unavailable: %s", exc)
            with self._lock:
                self._error = str(exc)
        except Exception:  # pragma: no cover - defensive guard for runtime failures
            LOG.exception("Unhandled exception in pipeline loop")
            with self._lock:
                self._error = "Pipeline crashed; check logs for details."
        finally:
            LOG.info("Person follower pipeline thread exiting")

    def _process(self, bundle: FrameBundle) -> PipelineState:
        detections = self._detector.detect(bundle.color_bgr)
        detection = detections[0] if detections else None
        estimate = self._estimator.estimate(detection, bundle.depth_m, bundle.intrinsics) if detection else None
        command = self._controller.compute(estimate)

        now = time.time()
        if self._last_timestamp is not None:
            dt = now - self._last_timestamp
            if dt > 0:
                inst_fps = 1.0 / dt
                if self._fps == 0.0:
                    self._fps = inst_fps
                else:
                    self._fps = 0.8 * self._fps + 0.2 * inst_fps
        self._last_timestamp = now

        annotated_bytes: bytes | None = None
        if self._annotate:
            annotated = bundle.color_bgr.copy()
            annotate_frame(annotated, detection, estimate)
            overlay_command(annotated, command)
            success, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                annotated_bytes = encoded.tobytes()

        return PipelineState(
            timestamp=now,
            detection=detection,
            estimate=estimate,
            command=command,
            annotated_frame_jpeg=annotated_bytes,
            fps=self._fps,
        )
