# Person Follower with Intel RealSense

Real-time person detection and following pipeline that fuses Intel RealSense RGB and depth data. The application detects people in the color stream, estimates their 3D position using depth, and outputs velocity commands suitable for a mobile robot to pursue the target while maintaining distance.

## Features

- Captures synchronized color and depth frames from Intel RealSense.
- Runs YOLO11 nano person detection (optimized for edge/Jetson) on the RGB stream.
- Projects detections into 3D using RealSense intrinsics and depth data.
- Computes proportional linear/angular velocity commands to follow the target.
- Optional video preview and recording with detection overlays.
- FastAPI-based web dashboard to monitor detections, depth estimates, and control commands.

## Requirements

- Intel RealSense depth camera (D4xx or L5xx series recommended).
- Intel RealSense SDK 2.0 (`librealsense`) with `pyrealsense2` Python bindings.
- Python 3.10+ with the dependencies listed in `requirements.txt`.
- GPU is optional; the detector defaults to CPU execution.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pyrealsense2` fails to install from PyPI, install the Intel RealSense SDK from https://github.com/IntelRealSense/librealsense following the instructions for your platform, then retry the `pip install`.

> **Note:** For Jetson or GPU acceleration, replace the default `onnxruntime` wheel with the platform-specific GPU build (e.g., `onnxruntime-gpu`).

## Usage

```bash
python -m src.main --display
```

Key options:

- `--model`: YOLO artifact (Ultralytics `.pt`, exported `.onnx`, or TensorRT `.engine`; default `yolo11n.pt`).
- `--device`: Torch device (`cpu`, `cuda:0`, ...).
- `--half`: Enable FP16 inference (recommended on Jetson / CUDA devices).
- `--target-distance`: Desired following distance in meters (default `1.2`).
- `--display`: Show annotated frames in an OpenCV window.
- `--save-video`: Record annotated output to an MP4 file.
- `--yellow-filter`: Require the tracked person to wear a yellow vest (HSV thresholds tunable with `--yellow-hue`, `--yellow-saturation-min`, `--yellow-value-min`, `--yellow-coverage`).

Run `python -m src.main --help` for the full list of arguments.

## Web Dashboard

Launch the FastAPI application and open the provided URL in your browser:

```bash
python -m src.web.server --host 0.0.0.0 --port 8000
```

The dashboard renders the latest annotated frame, distance/bearing estimates, and controller outputs. It polls REST endpoints exposed by the FastAPI backend (`/api/status`, `/api/frame`, `/api/health`). If the RealSense pipeline is unavailable, the UI surfaces the reported error.

- Install the Jetson-specific PyTorch wheels (or JetPack-provided ones) before installing `ultralytics` to ensure CUDA acceleration is enabled.
- Optionally install the Jetson-compatible ONNX Runtime (e.g., `onnxruntime-gpu`) if you plan to run exported `.onnx` models.
- Run the follower with `python -m src.main --device cuda:0 --half` (defaults to this combo automatically on Jetson/`aarch64`).
- When using the FastAPI dashboard with ONNX/TensorRT models, export `FOLLOWER_MODEL=/path/to/model` and `FOLLOWER_DEVICE=cuda:0`.
- Monitor thermals and power; consider locking Jetson clocks with `sudo jetson_clocks` for consistent performance.

## TensorRT Engine

Already exported a TensorRT engine (for example `yolo11n.engine`)? Point the follower at it and the detector will automatically use the Ultralytics TensorRT backend:

```bash
python -m src.main --model /path/to/yolo11n.engine --device cuda:0
```

The FastAPI server can do the same via environment variables:

```bash
FOLLOWER_MODEL=/path/to/yolo11n.engine FOLLOWER_DEVICE=cuda:0 python -m src.web.server
```

> TensorRT engines are specific to the GPU architecture and TensorRT version used at export time. Re-export if you update JetPack or move to different hardware.

The default configuration already uses the Ultralytics `.pt` weights. You can point to a custom `.pt` like so:

```bash
python -m src.main --model /models/yolo11n.pt --device cuda:0 --half
```

or for the web UI:

```bash
FOLLOWER_MODEL=/models/yolo11n.pt python -m src.web.server
```

## How It Works

1. `RealSenseCamera` streams aligned color and depth frames.
2. `PersonDetector` runs a YOLO11 nano model and returns the most confident `person` detection.
3. `TargetEstimator` samples depth around the detection and projects it into 3D camera space.
4. `SimpleFollowerController` computes linear/angular velocities to close the distance and center the person.
5. Status logs and optional overlays report the perceived target pose and command outputs.

## Integrating With a Robot

- Subscribe to the command output (`linear_velocity_mps`, `angular_velocity_radps`) and translate it to your robot's drive interface.
- Consider low-pass filtering or PID control in your robot stack for smoother motion.
- Ensure collision avoidance or safety layers override commands when necessary.

## Troubleshooting

- **No camera found**: Verify `realsense-viewer` detects the device and that udev rules are installed (Linux).
- **Detector download issues**: Copy your exported `yolo11n.onnx` model into the project or provide an absolute path with `--model`.
- **Depth spikes**: Adjust `--depth-kernel` and `--min-valid-ratio` to filter noise, or lower the stream resolution for cleaner depth.
- **Slow inference**: Switch to an even smaller model (e.g., `yolo11n-int8` if you export to TensorRT) or enable GPU with `--device cuda`.
