# ComfyUI-Kinetic-Motion 🎨✨

A powerful ComfyUI custom node suite for **high-fidelity 3D human pose tracking, kinetic motion curve extraction, dense optical flow velocity field generation, and expressive physical brushstroke rendering**.

Designed specifically for AI animation, video-to-video generative stylization, and abstract kinetic artwork pipelines.

---

## 🌟 Key Features

### 1. **Google Kinetic Motion Curve Extractor** (`KineticMotionCurveExtractor`)
* **Multi-Stage Kinematics Pipeline**:
  1. **Human Silhouette Segmentation**: Isolates dancing subjects with active contour highlighting.
  2. **MediaPipe 3D Pose Keypoints**: Tracks 33 anatomical joints with exponential moving average (EMA) temporal smoothing.
  3. **Dense Optical Flow Velocity Field**: Farnebäck optical flow computing instantaneous spatial velocity and direction.
  4. **Velocity-Modulated Spline Trajectories**: Translates joint trajectories into smooth Catmull-Rom or Bézier splines with adaptive width and luminescence based on velocity.
* **Kinematic Ribbon Sweeps**: Connects forearm, lower leg, shoulder, and pelvis kinematics.
* **4-Stage 2x2 HUD Comparison Grid**: Generates a real-time annotated diagnostic grid for inspecting all intermediate stages.
* **Zero-Setup Model Auto-Download**: Automatically fetches Google MediaPipe Pose Landmarker models (`lite`, `full`, `heavy`) on first run.

---

### 2. **Kinetic Motion-to-Brush Renderer** (`KineticMotionToBrushRenderer`)
* **Physical Paint Temporal Decay**: Canvas simulates wet paint persistence, fading gracefully according to configurable exponential decay.
* **Dynamic Impasto Stroke Modulation**: Modulates ribbon thickness and luminescence dynamically from instantaneous velocity ($v$) and acceleration bursts ($a$).
* **Hierarchical Anatomical Weighting**: Prioritizes expressive dance marks on wrists and feet while maintaining anchoring at elbows, knees, and torso.
* **Active Leading Brush Heads**: Renders luminous leading brush heads at $P(t)$ with speed-responsive radius scaling.
* **Secondary Momentum Streamlines**: Emits trailing optical flow particle streamers from moving limbs.
* **Luminescence Bloom Glow**: Soft Gaussian bloom surrounding active stroke ribbons.
* **Curated Color Modes**:
  - `luminous_white`: Pure radiant white/silver strokes on dark canvas.
  - `kinetic_spectrum`: Vibrant multi-colored joint palette (Rose, Cyan, Violet, Gold, Emerald).
  - `warm_amber`: Fiery sunset amber, gold, and vermilion tones.
  - `cool_cyan`: Oceanic cyan, deep blue, and indigo highlights.

---

### 3. **Kinetic Video Combine & Preview** (`KineticVideoCombine`)
* Combines frame batches into `.mp4`, animated `.webp`, or `.gif`.
* Renders instant inline animated previews directly in the ComfyUI node interface without requiring third-party video preview suites.

---

## 📦 Installation

### Option 1: Git Clone (Recommended)
Navigate to your ComfyUI `custom_nodes` directory and clone the repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/denga723/ComfyUI-Kinetic-Motion.git
cd ComfyUI-Kinetic-Motion
pip install -r requirements.txt
```

### Option 2: ComfyUI Manager
Search for `ComfyUI-Kinetic-Motion` in ComfyUI Manager and click **Install**.

---

## 📂 Included Workflows

The package includes pre-configured workflow JSON files in the `workflows/` directory:

1. **`kinetic_motion_and_brush_preview_suite_workflow.json`**:
   * Pure standalone workflow featuring the Extractor and Brush Renderer.
   * Full **7-output Preview Suite**:
     - Stage 1: Human Segmentation Mask
     - Stage 2: MediaPipe 3D Pose Keypoints
     - Stage 3: Dense Optical Flow Field
     - Stage 4: Converted Bezier Splines
     - 4-Stage 2x2 HUD Grid
     - Clean Motion Extractor Output
     - Kinetic Dynamic Brushstrokes Video

2. **`gemini_omni_kinetic_motion_to_brush_workflow.json`**:
   * Complete end-to-end video-to-video stylization pipeline connecting Kinetic Brushstrokes to Google Gemini Omni and 4 Style Reference Images.

3. **`kinetic_motion_parameters_comparison_workflow.json`**:
   * Parameter matrix comparing different decay rates, stroke thicknesses, velocity factors, and color modes.

---

## ⚙️ Node Parameters Reference

### `KineticMotionCurveExtractor`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `video_or_images` | `ANY` | - | Input video path or IMAGE batch tensor `[B, H, W, C]` |
| `spline_type` | `Combo` | `catmull_rom_spline` | Spline interpolation (`catmull_rom_spline`, `bezier_spline`, `linear`) |
| `trail_window` | `INT` | `20` | Temporal history window length in frames |
| `stroke_base_thickness` | `INT` | `18` | Base width of spline ribbons in pixels |
| `speed_to_width_factor` | `FLOAT` | `1.8` | Velocity modulation on stroke width |
| `speed_to_brightness_factor` | `FLOAT` | `1.2` | Velocity modulation on stroke luminescence |
| `dense_optical_flow` | `Combo` | `enable` | Compute dense optical flow vector field |
| `temporal_smoothing` | `FLOAT` | `0.6` | Pose landmark EMA smoothing factor (0.0 to 1.0) |
| `model_complexity` | `Combo` | `full` | MediaPipe model complexity (`lite`, `full`, `heavy`) |
| `fps` | `INT` | `24` | Target playback frame rate |

### `KineticMotionToBrushRenderer`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `motion_input` | `ANY` | - | `kinetic_motion_data` bundle from extractor |
| `trajectory_history_length`| `INT` | `24` | Trajectory temporal history length |
| `temporal_decay` | `FLOAT` | `0.88` | Canvas fade rate per frame (0.50 - 0.99) |
| `stroke_base_width` | `INT` | `22` | Base brush ribbon width |
| `min_stroke_width` | `INT` | `3` | Minimum tapered stroke width |
| `max_stroke_width` | `INT` | `48` | Maximum explosive stroke width |
| `velocity_influence` | `FLOAT` | `1.8` | Velocity modulation on stroke energy |
| `acceleration_influence` | `FLOAT` | `0.8` | Acceleration burst scaling on directional shifts |
| `brush_head_size` | `INT` | `14` | Radius of active leading brush head |
| `optical_flow_strength` | `FLOAT` | `0.4` | Intensity of optical flow streamers |
| `glow_strength` | `FLOAT` | `0.6` | Soft luminescence bloom intensity |
| `brush_color_mode` | `Combo` | `luminous_white` | Palette preset (`luminous_white`, `kinetic_spectrum`, `warm_amber`, `cool_cyan`) |

---

## 🛠️ Requirements

- Python >= 3.9
- `torch >= 2.0.0`
- `mediapipe >= 0.10.0`
- `opencv-python >= 4.7.0`
- `numpy >= 1.22.0`
- `Pillow >= 9.0.0`

---

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## 👤 Author

- **Andrew Deng** ([denga723@newschool.edu](mailto:denga723@newschool.edu))
