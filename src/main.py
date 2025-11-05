from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from .person_following.camera import FrameBundle, RealSenseCamera, RealSenseUnavailableError
from .person_following.config import (
    CameraConfig,
    ControllerConfig,
    DetectorConfig,
    FollowerConfig,
)
from .person_following.controller import ControlCommand, SimpleFollowerController
from .person_following.detector import Detection, PersonDetector
from .person_following.tracker import TargetEstimate, TargetEstimator
from .person_following.visualization import annotate_frame, overlay_command

LOG = logging.getLogger("person_follower")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time person following with Intel RealSense.")

    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frames per second.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model artifact (.pt, .onnx, .engine) or Ultralytics hub name.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu, cuda:0, ...).")
    parser.add_argument("--half", action="store_true", help="Enable half precision inference (FP16).")
    parser.add_argument("--target-distance", type=float, default=1.2, help="Desired following distance in meters.")
    parser.add_argument("--distance-gain", type=float, default=0.8, help="Proportional gain for distance control.")
    parser.add_argument("--angle-gain", type=float, default=2.2, help="Proportional gain for angular control.")
    parser.add_argument("--max-linear", type=float, default=0.6, help="Maximum linear speed (m/s).")
    parser.add_argument("--max-angular", type=float, default=1.5, help="Maximum angular speed (rad/s).")
    parser.add_argument("--depth-kernel", type=int, default=5, help="Depth averaging kernel size (odd integer).")
    parser.add_argument("--min-valid-ratio", type=float, default=0.3, help="Minimum ratio of valid depth samples.")
    parser.add_argument("--display", action="store_true", help="Render annotated frames in an OpenCV window.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional output video path (mp4).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...).")
    parser.add_argument("--yellow-filter", action="store_true", help="Require detected person to wear a yellow vest.")
    parser.add_argument("--yellow-hue", type=int, nargs=2, metavar=("MIN", "MAX"), default=(20, 35), help="HSV hue range for yellow vest filtering.")
    parser.add_argument("--yellow-saturation-min", type=int, default=100, help="Minimum HSV saturation for yellow vest filtering.")
    parser.add_argument("--yellow-value-min", type=int, default=80, help="Minimum HSV value for yellow vest filtering.")
    parser.add_argument("--yellow-coverage", type=float, default=0.15, help="Minimum coverage ratio of yellow pixels inside the bounding box.")

    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_config(args: argparse.Namespace) -> FollowerConfig:
    camera = CameraConfig(width=args.width, height=args.height, fps=args.fps)

    device = args.device
    use_half = args.half

    if device is None:
        try:
            import torch  # type: ignore[import]
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            device = "cuda:0"
            if platform.machine().lower() == "aarch64" and not use_half:
                use_half = True

    if use_half and (device is None or device.startswith("cpu")):
        LOG.warning("Disabling half precision because no CUDA device is configured.")
        use_half = False

    detector = DetectorConfig(
        model_name=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=device,
        use_half_precision=use_half,
        enable_yellow_filter=args.yellow_filter,
        yellow_hue_min=args.yellow_hue[0],
        yellow_hue_max=args.yellow_hue[1],
        yellow_saturation_min=args.yellow_saturation_min,
        yellow_value_min=args.yellow_value_min,
        yellow_coverage_threshold=args.yellow_coverage,
    )
    controller = ControllerConfig(
        target_distance_m=args.target_distance,
        distance_gain=args.distance_gain,
        angle_gain=args.angle_gain,
        max_linear_speed_mps=args.max_linear,
        max_angular_speed_radps=args.max_angular,
    )
    return FollowerConfig(camera=camera, detector=detector, controller=controller)


def create_video_writer(path: Path, width: int, height: int, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = create_config(args)

    try:
        camera = RealSenseCamera(config.camera)
    except RealSenseUnavailableError as exc:
        LOG.error("%s", exc)
        return 1

    detector = PersonDetector(config.detector)
    estimator = TargetEstimator(depth_kernel=args.depth_kernel, min_valid_ratio=args.min_valid_ratio)
    controller = SimpleFollowerController(config.controller)

    video_writer = None
    if args.save_video is not None:
        video_writer = create_video_writer(args.save_video, config.camera.width, config.camera.height, config.camera.fps)
        LOG.info("Recording annotated stream to %s", args.save_video)

    last_log_time = 0.0

    try:
        with camera.streaming():
            while True:
                frame_bundle = camera.frames()
                if frame_bundle is None:
                    continue
                detection, estimate, command = process_frame(
                    frame_bundle, detector=detector, estimator=estimator, controller=controller
                )
                now = time.time()
                if now - last_log_time >= 0.5:
                    log_status(detection, estimate, command)
                    last_log_time = now

                if args.display or video_writer is not None:
                    viz = frame_bundle.color_bgr.copy()
                    annotate_frame(viz, detection, estimate)
                    overlay_command(viz, command)

                    if args.display:
                        cv2.imshow("PersonFollower", viz)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            LOG.info("Received quit signal (q).")
                            break
                    if video_writer is not None:
                        video_writer.write(viz)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
    finally:
        if video_writer is not None:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


def process_frame(
    bundle: FrameBundle,
    *,
    detector: PersonDetector,
    estimator: TargetEstimator,
    controller: SimpleFollowerController,
) -> tuple[Optional[Detection], Optional[TargetEstimate], ControlCommand]:
    detections = detector.detect(bundle.color_bgr)
    detection = detections[0] if detections else None
    estimate = estimator.estimate(detection, bundle.depth_m, bundle.intrinsics) if detection else None
    command = controller.compute(estimate)
    return detection, estimate, command


def log_status(
    detection: Optional[Detection],
    estimate: Optional[TargetEstimate],
    command: ControlCommand,
) -> None:
    if detection is None or estimate is None:
        LOG.info("Searching for person...")
        return

    LOG.info(
        "distance=%.2fm bearing=%+.1fdeg lin=%.2fm/s ang=%+.2frad/s depth_valid=%.0f%%",
        estimate.distance_m,
        estimate.bearing_deg,
        command.linear_velocity_mps,
        command.angular_velocity_radps,
        estimate.valid_depth_ratio * 100.0,
    )


if __name__ == "__main__":
    sys.exit(main())
