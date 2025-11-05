from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from .config import DetectorConfig
LOG = logging.getLogger(__name__)

@dataclass
class Detection:
    """Bounding box detection in pixel space."""

    xyxy: tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.xyxy
        return int(x2 - x1)

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.xyxy
        return int(y2 - y1)

class PersonDetector:
    """Person detector helper supporting Ultralytics PyTorch and ONNXRuntime backends."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        model_name = self._config.model_name.lower()

        if model_name.endswith(".onnx"):
            self._backend = _OnnxRuntimeDetector(self._config)
        else:
            self._backend = _UltralyticsDetector(self._config)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr.ndim != 3:
            raise ValueError("Expected 3-channel color frame for detection.")
        return self._backend.detect(frame_bgr)


class _UltralyticsDetector:
    """Wrapper around ultralytics.YOLO for backward compatibility."""

    def __init__(self, config: DetectorConfig) -> None:
        model_name = config.model_name.lower()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - requires runtime dependency
            raise RuntimeError(
                "ultralytics package is required for person detection with PyTorch weights. "
                "Install it with `pip install ultralytics` or provide an ONNX model."
            ) from exc

        self._config = config
        self._is_tensorrt = model_name.endswith(".engine")

        if self._is_tensorrt and self._config.use_half_precision:
            LOG.info("TensorRT engine already optimized; ignoring `use_half_precision` flag.")
            self._config.use_half_precision = False

        if self._config.use_half_precision and not _supports_half(self._config.device):
            LOG.warning(
                "Disabling half precision because device %s does not support FP16.",
                self._config.device or "cpu",
            )
            self._config.use_half_precision = False

        LOG.info(
            "Loading YOLO model %s on device %s (half=%s)",
            self._config.model_name,
            self._config.device or "auto",
            self._config.use_half_precision,
        )
        self._model = YOLO(self._config.model_name)
        names = getattr(self._model, "names", None)
        if names is None:
            model_attr = getattr(self._model, "model", None)
            names = getattr(model_attr, "names", None) if model_attr is not None else None
        if names is None:
            names = {0: "person"}
        self._names = names
        self._tracker_config = self._config.tracker_config or "bytetrack.yaml"
        self._target_track_id: Optional[int] = None

    def _is_person(self, class_id: int) -> bool:
        if class_id == self._config.person_class_id:
            return True
        if isinstance(self._names, dict):
            return self._names.get(class_id) == "person"
        if isinstance(self._names, (list, tuple)):
            if 0 <= class_id < len(self._names):
                return self._names[class_id] == "person"
        return False

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        track_results = self._model.track(  # type: ignore[arg-type]
            frame_bgr,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            verbose=False,
            device=self._config.device,
            half=self._config.use_half_precision,
            persist=True,
            stream=False,
            tracker=self._tracker_config,
        )

        if not track_results:
            self._target_track_id = None
            return []

        result = track_results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self._target_track_id = None
            return []

        height, width = frame_bgr.shape[:2]
        candidates: list[tuple[float, tuple[int, int, int, int], Optional[int]]] = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if not self._is_person(cls_id):
                continue
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1f, y1f, x2f, y2f = xyxy
            x1 = int(np.clip(x1f, 0, width - 1))
            y1 = int(np.clip(y1f, 0, height - 1))
            x2 = int(np.clip(x2f, 0, width - 1))
            y2 = int(np.clip(y2f, 0, height - 1))
            track_tensor = getattr(box, "id", None)
            track_id: Optional[int] = None
            if track_tensor is not None:
                if hasattr(track_tensor, "detach"):
                    track_array = track_tensor.detach().cpu().numpy().reshape(-1)
                else:
                    track_array = np.array(track_tensor, copy=False).reshape(-1)
                if track_array.size > 0:
                    track_id = int(track_array[0])
            if self._config.enable_yellow_filter and not _passes_yellow_filter(
                frame_bgr, (x1, y1, x2, y2), self._config
            ):
                continue
            candidates.append((conf, (x1, y1, x2, y2), track_id))

        if not candidates:
            self._target_track_id = None
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected_conf: float
        selected_box: tuple[int, int, int, int]
        selected_track: Optional[int]

        if self._target_track_id is not None:
            match = next((c for c in candidates if c[2] == self._target_track_id), None)
            if match is not None:
                selected_conf, selected_box, selected_track = match
            else:
                selected_conf, selected_box, selected_track = candidates[0]
                self._target_track_id = selected_track
        else:
            selected_conf, selected_box, selected_track = candidates[0]
            self._target_track_id = selected_track

        backend_name = "TensorRT" if self._is_tensorrt else "Ultralytics"
        LOG.debug(
            "Tracking person with %s backend (track_id=%s confidence=%.3f)",
            backend_name,
            selected_track,
            selected_conf,
        )
        return [Detection(xyxy=selected_box, confidence=selected_conf, track_id=selected_track)]


class _OnnxRuntimeDetector:
    """Minimal ONNXRuntime inference pipeline for YOLO models."""

    def __init__(self, config: DetectorConfig) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - requires runtime dependency
            raise RuntimeError(
                "onnxruntime is required for ONNX inference. Install it with `pip install onnxruntime` "
                "or the GPU variant suitable for your platform."
            ) from exc

        self._config = config
        providers = self._select_providers(ort)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # reduce logging noise

        LOG.info(
            "Loading ONNX model %s with providers %s",
            self._config.model_name,
            providers,
        )
        self._session = ort.InferenceSession(
            self._config.model_name,
            providers=providers,
            sess_options=session_options,
        )

        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape
        # Support dynamic axes (-1)
        if isinstance(shape[2], int) and shape[2] > 0:
            self._input_height = shape[2]
        else:
            self._input_height = self._config.input_size
        if isinstance(shape[3], int) and shape[3] > 0:
            self._input_width = shape[3]
        else:
            self._input_width = self._config.input_size
        self._output_names = [out.name for out in self._session.get_outputs()]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        image, ratio, pad_left, pad_top = self._preprocess(frame_bgr)
        outputs = self._session.run(self._output_names, {self._input_name: image})
        predictions = outputs[0]

        if predictions.ndim == 3:
            predictions = np.squeeze(predictions, axis=0)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = predictions[:, :4]
        if predictions.shape[1] > 5:
            obj_conf = predictions[:, 4]
            class_scores = predictions[:, 5:]
            if class_scores.shape[1] == 0:
                confidences = obj_conf
            else:
                person_idx = min(self._config.person_class_id, class_scores.shape[1] - 1)
                confidences = obj_conf * class_scores[:, person_idx]
        else:
            confidences = predictions[:, 4]

        mask = confidences >= self._config.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        if boxes.size == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes, ratio, pad_left, pad_top, frame_bgr.shape[1], frame_bgr.shape[0])
        keep_indices = _nms(boxes_xyxy, confidences, self._config.iou_threshold, self._config.max_detections)

        detections: list[Detection] = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            detections.append(
                Detection(
                    xyxy=(
                        int(np.clip(x1, 0, frame_bgr.shape[1] - 1)),
                        int(np.clip(y1, 0, frame_bgr.shape[0] - 1)),
                        int(np.clip(x2, 0, frame_bgr.shape[1] - 1)),
                        int(np.clip(y2, 0, frame_bgr.shape[0] - 1)),
                    ),
                    confidence=float(confidences[idx]),
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        LOG.debug("Detected %d person candidates with ONNX backend", len(detections))
        return detections

    def _select_providers(self, ort_module) -> list[str]:
        available = ort_module.get_available_providers()
        if self._config.device and self._config.device.lower().startswith("cuda"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            LOG.warning("CUDAExecutionProvider not available; falling back to CPUExecutionProvider.")
        return ["CPUExecutionProvider"]

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame_bgr.shape[:2]
        scale = min(self._input_width / w, self._input_height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self._input_height, self._input_width, 3), 114, dtype=np.uint8)
        top = (self._input_height - new_h) // 2
        left = (self._input_width - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale, left, top

    def _xywh_to_xyxy(
        self,
        boxes_xywh: np.ndarray,
        scale: float,
        pad_left: int,
        pad_top: int,
        original_w: int,
        original_h: int,
    ) -> np.ndarray:
        cx = (boxes_xywh[:, 0] - pad_left) / scale
        cy = (boxes_xywh[:, 1] - pad_top) / scale
        w = boxes_xywh[:, 2] / scale
        h = boxes_xywh[:, 3] / scale

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack((x1, y1, x2, y2), axis=1)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_h - 1)
        return boxes


def _supports_half(device: str | None) -> bool:
    if device is None:
        return False
    return not device.lower().startswith("cpu")


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float, max_detections: int) -> list[int]:

    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if len(keep) >= max_detections:
            break

        rest = order[1:]
        if rest.size == 0:
            break

        ious = _iou(boxes[idx], boxes[rest])
        order = rest[ious <= iou_thresh]

    return keep


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter_area
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter_area / union


def _passes_yellow_filter(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    config: DetectorConfig,
) -> bool:
    x1, y1, x2, y2 = box
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [
            config.yellow_hue_min,
            config.yellow_saturation_min,
            config.yellow_value_min,
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            config.yellow_hue_max,
            255,
            255,
        ],
        dtype=np.uint8,
    )

    mask = cv2.inRange(hsv, lower, upper)
    coverage = float(np.count_nonzero(mask)) / float(mask.size)

    LOG.debug("Yellow coverage=%.3f threshold=%.3f", coverage, config.yellow_coverage_threshold)
    return coverage >= config.yellow_coverage_threshold
