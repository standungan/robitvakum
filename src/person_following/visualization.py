from __future__ import annotations

import cv2

from .controller import ControlCommand
from .detector import Detection
from .tracker import TargetEstimate


def annotate_frame(frame_bgr, detection: Detection | None, estimate: TargetEstimate | None) -> None:
    if detection is None:
        cv2.putText(frame_bgr, "No person detected", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    x1, y1, x2, y2 = detection.xyxy
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if estimate:
        text = f"{estimate.distance_m:.2f}m | {estimate.bearing_deg:+.1f}deg"
        cv2.putText(frame_bgr, text, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def overlay_command(frame_bgr, command: ControlCommand) -> None:
    text = f"v={command.linear_velocity_mps:+.2f} m/s | w={command.angular_velocity_radps:+.2f} rad/s"
    cv2.putText(frame_bgr, text, (16, frame_bgr.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

