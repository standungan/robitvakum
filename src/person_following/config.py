from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    """Streaming configuration for the RealSense camera."""

    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = True
    enable_color: bool = True

@dataclass
class DetectorConfig:
    """Person detector runtime parameters."""

    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.5
    device: str | None = "dla:0"
    use_half_precision: bool = True
    input_size: int = 640
    max_detections: int = 5
    person_class_id: int = 0
    tracker_config: str | None = "bytetrack.yaml"
    enable_yellow_filter: bool = False
    yellow_hue_min: int = 10
    yellow_hue_max: int = 45
    yellow_saturation_min: int = 100
    yellow_value_min: int = 80
    yellow_coverage_threshold: float = 0.15

@dataclass
class ControllerConfig:
    """Simple proportional controller parameters for robot commands."""
    target_distance_m: float = 2.5
    distance_gain: float = 0.8
    angle_gain: float = 2.2
    max_linear_speed_mps: float = 0.6
    max_angular_speed_radps: float = 1.5

@dataclass
class FollowerConfig:
    """Aggregate configuration for the person follower pipeline."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
