"""
ComfyUI-Kinetic-Motion
======================
High-precision kinetic motion curve extraction, MediaPipe 3D pose tracking,
dense optical flow kinematics, and expressive physical brushstroke rendering for ComfyUI.

Nodes included:
- Google Kinetic Motion Curve Extractor (KineticMotionCurveExtractor)
- Kinetic Motion-to-Brush Renderer (KineticMotionToBrushRenderer)
- Kinetic Video Combine & Preview (KineticVideoCombine)
"""

import os
import urllib.request
import tempfile
from typing import List, Any, Optional, Dict, Union
import numpy as np
from PIL import Image
import cv2
import torch

try:
    import folder_paths
except ImportError:
    class DummyFolderPaths:
        base_path = os.getcwd()
        def get_output_directory(self):
            out = os.path.join(self.base_path, "output")
            os.makedirs(out, exist_ok=True)
            return out
        def get_temp_directory(self):
            tmp = os.path.join(self.base_path, "temp")
            os.makedirs(tmp, exist_ok=True)
            return tmp
        def get_save_image_path(self, prefix, outdir, w, h):
            os.makedirs(outdir, exist_ok=True)
            return outdir, prefix, 1, "", prefix
    folder_paths = DummyFolderPaths()


class AnyType(str):
    """ComfyUI wildcard type to allow connections to/from any socket."""
    def __ne__(self, __value: object) -> bool:
        return False

ANY_TYPE = AnyType("*")


def get_video_file_path(val: Any) -> Optional[str]:
    """Helper to extract a video file path from various ComfyUI video loader outputs."""
    if val is None:
        return None
    if isinstance(val, str) and (val.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv", ".gif")) or os.path.exists(val.strip('\'"'))):
        return val.strip('\'"')
    if hasattr(val, "get_stream_source") and callable(val.get_stream_source):
        try:
            src = val.get_stream_source()
            if isinstance(src, str) and (os.path.exists(src) or src.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))):
                return src
        except Exception:
            pass
    if hasattr(val, "_VideoFromFile__file"):
        f = getattr(val, "_VideoFromFile__file")
        if isinstance(f, str) and (os.path.exists(f) or f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))):
            return f
    for attr in ["file", "path", "file_path", "video_path", "filename", "video"]:
        if hasattr(val, attr) and isinstance(getattr(val, attr), str):
            f = getattr(val, attr)
            if os.path.exists(f) or f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
                return f
    return None


class KineticMotionCurveExtractor:
    """
    Extracts multi-stage kinetic motion curves, MediaPipe 3D pose landmarks,
    human silhouette segmentation, and dense optical flow fields from video input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_or_images": (ANY_TYPE, {"tooltip": "Video file path, video loader output, or IMAGE batch tensor [B, H, W, C]"}),
            },
            "optional": {
                "spline_type": (["catmull_rom_spline", "bezier_spline", "linear"], {"default": "catmull_rom_spline", "tooltip": "Mathematical interpolation method for trajectory ribbons"}),
                "trail_window": ("INT", {"default": 20, "min": 2, "max": 60, "step": 1, "tooltip": "Temporal trail history window length (frames)"}),
                "stroke_base_thickness": ("INT", {"default": 18, "min": 2, "max": 50, "step": 1, "tooltip": "Base thickness of spline ribbons"}),
                "speed_to_width_factor": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Width modulation scaling based on instantaneous joint velocity"}),
                "speed_to_brightness_factor": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Luminescence scaling based on instantaneous joint velocity"}),
                "dense_optical_flow": (["enable", "disable"], {"default": "enable", "tooltip": "Compute dense optical flow vector field and streamlines"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "EMA smoothing factor for pose landmark jitter suppression"}),
                "model_complexity": (["full", "heavy", "lite"], {"default": "full", "tooltip": "MediaPipe Pose Landmarker model complexity"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "Target frame rate"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "KINETIC_MOTION_DATA")
    RETURN_NAMES = (
        "motion_representation",
        "stage_pipeline_grid",
        "1_segmentation_mask",
        "2_pose_keypoints",
        "3_optical_flow",
        "4_bezier_splines",
        "motion_video_file",
        "kinetic_motion_data"
    )
    FUNCTION = "extract_motion_representation"
    CATEGORY = "KineticMotion"
    DESCRIPTION = "Extracts 3D pose kinematics, segmentation, optical flow, and velocity-modulated spline curves."

    def _catmull_rom(self, pts, num_samples=8):
        if len(pts) < 2: return pts
        if len(pts) == 2: return pts
        pts = [pts[0]] + list(pts) + [pts[-1]]
        curve = []
        for i in range(len(pts) - 3):
            p0 = np.array(pts[i], dtype=float)
            p1 = np.array(pts[i+1], dtype=float)
            p2 = np.array(pts[i+2], dtype=float)
            p3 = np.array(pts[i+3], dtype=float)
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3)
                curve.append((int(round(pt[0])), int(round(pt[1]))))
        curve.append(pts[-2])
        return curve

    def _draw_hud(self, img, text, subtext=""):
        h, w, _ = img.shape
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (12, 14, 18), -1)
        cv2.line(overlay, (0, 36), (w, 36), (0, 230, 255), 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 1, cv2.LINE_AA)
        if subtext:
            tw = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            cv2.putText(img, subtext, (max(10, w - tw - 10), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 230), 1, cv2.LINE_AA)

    def extract_motion_representation(
        self,
        video_or_images,
        spline_type="catmull_rom_spline",
        trail_window=20,
        stroke_base_thickness=18,
        speed_to_width_factor=1.8,
        speed_to_brightness_factor=1.2,
        dense_optical_flow="enable",
        temporal_smoothing=0.6,
        model_complexity="full",
        fps=24
    ):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, PoseLandmarksConnections

        model_dir = os.path.join(folder_paths.base_path, "models", "mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"pose_landmarker_{model_complexity}.task")
        
        if not os.path.exists(model_path):
            urls = {
                "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            }
            url = urls.get(model_complexity, urls["full"])
            print(f"[KineticMotion] Downloading {model_complexity} pose model from {url}...")
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        frames_rgb = []
        v_path = get_video_file_path(video_or_images)
        
        if v_path and os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps and vid_fps > 0:
                fps = int(round(vid_fps))
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        elif isinstance(video_or_images, torch.Tensor):
            t = video_or_images
            if len(t.shape) == 3: t = t.unsqueeze(0)
            np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for i in range(np_frames.shape[0]):
                frames_rgb.append(np_frames[i])
        elif isinstance(video_or_images, list):
            for item in video_or_images:
                if isinstance(item, torch.Tensor):
                    t = item
                    if len(t.shape) == 3: t = t.unsqueeze(0)
                    np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    for i in range(np_frames.shape[0]):
                        frames_rgb.append(np_frames[i])

        if not frames_rgb:
            raise ValueError("[KineticMotionCurveExtractor] No frames could be decoded from input.")

        TRAIL_PALETTE = {
            15: (255, 60, 140),   # Left wrist (Rose)
            16: (0, 240, 255),    # Right wrist (Cyan)
            13: (255, 170, 30),   # Left elbow (Amber)
            14: (50, 255, 120),   # Right elbow (Emerald)
            27: (170, 70, 255),   # Left ankle (Violet)
            28: (255, 230, 20),   # Right ankle (Gold)
            0:  (240, 250, 255),  # Head (Luminous White)
            11: (100, 180, 255),  # Left shoulder
            12: (100, 180, 255),  # Right shoulder
            23: (255, 120, 180),  # Left hip
            24: (255, 120, 180),  # Right hip
        }

        KINEMATIC_SWEEPS = [
            (15, 13), (16, 14), # Forearms
            (27, 25), (28, 26), # Lower legs
            (11, 12), (23, 24), # Shoulders & Pelvis
        ]

        smoothed_landmarks = {}
        history = []
        history_vels = []
        history_accels = []
        masks_list = []
        flow_list = []
        flow_particles = []
        prev_gray = None

        final_frames = []
        grid_frames = []
        stage1_frames = []
        stage2_frames = []
        stage3_frames = []
        stage4_frames = []

        h, w, _ = frames_rgb[0].shape

        for frame_idx, rgb_frame in enumerate(frames_rgb):
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_img)

            # Stage 1: Human Segmentation Mask
            s1_canvas = np.zeros_like(rgb_frame)
            mask_bool = None
            if detection_result.segmentation_masks and len(detection_result.segmentation_masks) > 0:
                mask = detection_result.segmentation_masks[0].numpy_view()
                mask_bool = (mask.squeeze() > 0.4)
                mask_3c = np.repeat(mask, 3, axis=2)
                s1_canvas = (rgb_frame * mask_3c).astype(np.uint8)
                mask_u8 = (mask.squeeze() * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(s1_canvas, contours, -1, (0, 230, 255), 2)
            masks_list.append(mask_bool)
            s1_annotated = s1_canvas.copy()
            self._draw_hud(s1_annotated, "1. Human Segmentation", "Silhouette Isolated")
            stage1_frames.append(s1_annotated)

            # Stage 2: MediaPipe 3D Pose Keypoints
            curr_pts = {}
            s2_canvas = np.zeros_like(rgb_frame)
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                lm = detection_result.pose_landmarks[0]
                for i in range(len(lm)):
                    raw_pt = np.array([lm[i].x * w, lm[i].y * h], dtype=float)
                    if i in smoothed_landmarks and temporal_smoothing > 0:
                        alpha = 1.0 - np.clip(temporal_smoothing, 0.0, 0.95)
                        smoothed_pt = alpha * raw_pt + (1.0 - alpha) * smoothed_landmarks[i]
                    else:
                        smoothed_pt = raw_pt
                    smoothed_landmarks[i] = smoothed_pt
                    curr_pts[i] = (int(round(smoothed_pt[0])), int(round(smoothed_pt[1])))

                drawing_utils.draw_landmarks(
                    s2_canvas,
                    lm,
                    PoseLandmarksConnections.POSE_LANDMARKS,
                    drawing_styles.get_default_pose_landmarks_style()
                )

            history.append(curr_pts)

            # Instantaneous Velocity & Acceleration
            curr_vels = {}
            curr_accels = {}
            if len(history) >= 2:
                prev_p = history[-2]
                for k, pt in curr_pts.items():
                    if k in prev_p:
                        v = float(np.hypot(pt[0] - prev_p[k][0], pt[1] - prev_p[k][1]))
                        curr_vels[k] = v
                        if len(history_vels) > 0 and k in history_vels[-1]:
                            curr_accels[k] = abs(v - history_vels[-1][k])
            history_vels.append(curr_vels)
            history_accels.append(curr_accels)

            recent_history = history[-max(2, trail_window):]

            s2_annotated = s2_canvas.copy()
            self._draw_hud(s2_annotated, "2. MediaPipe Pose Keypoints", "33 3D Joints Tracked")
            stage2_frames.append(s2_annotated)

            # Stage 3: Dense Optical Flow Velocity
            s3_canvas = np.zeros_like(rgb_frame)
            flow = None
            if dense_optical_flow == "enable" and prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(rgb_frame)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                s3_canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_list.append(flow)
            prev_gray = gray
            s3_annotated = s3_canvas.copy()
            self._draw_hud(s3_annotated, "3. Dense Optical Flow", "Direction & Velocity Field")
            stage3_frames.append(s3_annotated)

            # Stage 4: Converted Bezier / Spline Curves
            s4_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Optical flow particle streamlines
            if flow is not None and curr_pts:
                for pt_idx in [15, 16, 27, 28, 0, 13, 14]:
                    if pt_idx in curr_pts:
                        px, py = curr_pts[pt_idx]
                        if 0 <= px < w and 0 <= py < h:
                            fx, fy = flow[py, px]
                            if np.hypot(fx, fy) > 1.5:
                                color = TRAIL_PALETTE.get(pt_idx, (200, 200, 255))
                                flow_particles.append([float(px), float(py), float(fx), float(fy), 0, color])
                
                new_particles = []
                for p in flow_particles:
                    px, py, fx, fy, age, color = p
                    if age < 8 and 0 <= int(px) < w and 0 <= int(py) < h:
                        npx, npy = px + fx * 1.5, py + fy * 1.5
                        alpha = max(0.1, 1.0 - (age / 8.0))
                        c_p = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                        cv2.line(s4_canvas, (int(px), int(py)), (int(npx), int(npy)), c_p, int(max(1, 4 * alpha)), cv2.LINE_AA)
                        new_particles.append([npx, npy, fx * 0.9, fy * 0.9, age + 1, color])
                flow_particles = new_particles[-150:]

            # Spline Curves with Velocity Mapping
            for pt_idx, base_color in TRAIL_PALETTE.items():
                raw_pts = [h_dict[pt_idx] for h_dict in recent_history if pt_idx in h_dict]
                if len(raw_pts) >= 2:
                    if spline_type == "catmull_rom_spline":
                        spline_pts = self._catmull_rom(raw_pts, num_samples=6)
                    else:
                        spline_pts = raw_pts

                    if len(spline_pts) >= 2:
                        speeds = []
                        for s_i in range(len(spline_pts) - 1):
                            dx = spline_pts[s_i+1][0] - spline_pts[s_i][0]
                            dy = spline_pts[s_i+1][1] - spline_pts[s_i][1]
                            speeds.append(np.hypot(dx, dy))
                        avg_spd = max(0.1, np.mean(speeds))

                        for s_i in range(len(spline_pts) - 1):
                            progress = (s_i + 1) / len(spline_pts)
                            local_spd = speeds[s_i]
                            
                            spd_factor = np.clip(local_spd / (avg_spd + 1e-5), 0.5, 3.0) if speed_to_width_factor > 0 else 1.0
                            thickness = int(max(2, round(progress * stroke_base_thickness * (1.0 + (speed_to_width_factor - 1.0) * (spd_factor - 1.0)))))
                            
                            brightness = np.clip(progress * (0.8 + 0.4 * (spd_factor if speed_to_brightness_factor > 0 else 1.0)), 0.1, 1.0)
                            seg_color = (
                                int(np.clip(base_color[0] * brightness, 0, 255)),
                                int(np.clip(base_color[1] * brightness, 0, 255)),
                                int(np.clip(base_color[2] * brightness, 0, 255))
                            )
                            cv2.line(s4_canvas, spline_pts[s_i], spline_pts[s_i+1], seg_color, thickness, cv2.LINE_AA)
                            cv2.circle(s4_canvas, spline_pts[s_i+1], int(thickness // 2), seg_color, -1, cv2.LINE_AA)

            # Kinematic Ribbon Sweeps
            if curr_pts:
                for (p1, p2) in KINEMATIC_SWEEPS:
                    if p1 in curr_pts and p2 in curr_pts:
                        pt1, pt2 = curr_pts[p1], curr_pts[p2]
                        cv2.line(s4_canvas, pt1, pt2, (220, 235, 255), 8, cv2.LINE_AA)
                        cv2.line(s4_canvas, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)

            final_frames.append(s4_canvas)

            # Annotated Stage 4 for preview
            s4_annotated = s4_canvas.copy()
            self._draw_hud(s4_annotated, "4. Converted Splines", "Velocity -> Stroke Width")
            stage4_frames.append(s4_annotated)

            # 2x2 Comparison Grid
            half_w, half_h = w // 2, h // 2
            r1 = np.hstack([cv2.resize(s1_annotated, (half_w, half_h)), cv2.resize(s2_annotated, (half_w, half_h))])
            r2 = np.hstack([cv2.resize(s3_annotated, (half_w, half_h)), cv2.resize(s4_annotated, (half_w, half_h))])
            grid_canvas = np.vstack([r1, r2])
            grid_frames.append(grid_canvas)

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for rf in final_frames:
            if rf.shape[1] != w or rf.shape[0] != h:
                rf = cv2.resize(rf, (w, h))
            out_writer.write(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        t_final = torch.from_numpy(np.stack(final_frames, axis=0).astype(np.float32) / 255.0)
        t_grid = torch.from_numpy(np.stack(grid_frames, axis=0).astype(np.float32) / 255.0)
        t_s1 = torch.from_numpy(np.stack(stage1_frames, axis=0).astype(np.float32) / 255.0)
        t_s2 = torch.from_numpy(np.stack(stage2_frames, axis=0).astype(np.float32) / 255.0)
        t_s3 = torch.from_numpy(np.stack(stage3_frames, axis=0).astype(np.float32) / 255.0)
        t_s4 = torch.from_numpy(np.stack(stage4_frames, axis=0).astype(np.float32) / 255.0)

        # Kinetic Motion Data Bundle for KineticMotionToBrushRenderer
        motion_data = {
            "frames_rgb": frames_rgb,
            "history_pts": history,
            "history_vels": history_vels,
            "history_accels": history_accels,
            "flow_list": flow_list,
            "masks_list": masks_list,
            "fps": fps,
            "width": w,
            "height": h,
            "num_frames": len(frames_rgb)
        }

        print(f"[KineticMotionCurveExtractor] Rendered {len(final_frames)} multi-stage frames @ {fps}fps to {tmp_mp4_path}")
        return (t_final, t_grid, t_s1, t_s2, t_s3, t_s4, tmp_mp4_path, motion_data)


class KineticMotionToBrushRenderer:
    """
    Transforms kinetic trajectory data and optical flow vectors into progressive,
    expressive abstract brushstrokes on canvas with physical paint temporal decay and bloom glow.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_input": (ANY_TYPE, {"tooltip": "Kinetic Motion Data from Google Kinetic Motion Extractor or video tensor"}),
            },
            "optional": {
                "trajectory_history_length": ("INT", {"default": 24, "min": 2, "max": 120, "step": 1, "tooltip": "Progressive trajectory temporal history length (frames)"}),
                "temporal_decay": ("FLOAT", {"default": 0.88, "min": 0.50, "max": 0.99, "step": 0.01, "tooltip": "Canvas decay rate per frame (0.88 = smooth physical paint fade)"}),
                "stroke_base_width": ("INT", {"default": 22, "min": 2, "max": 80, "step": 1, "tooltip": "Base brush ribbon stroke width"}),
                "min_stroke_width": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1, "tooltip": "Minimum tapered stroke width"}),
                "max_stroke_width": ("INT", {"default": 48, "min": 4, "max": 120, "step": 1, "tooltip": "Maximum explosive stroke width"}),
                "velocity_influence": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Velocity modulation on stroke width and energy"}),
                "acceleration_influence": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Acceleration burst scaling on sharp directional shifts"}),
                "brush_head_size": ("INT", {"default": 14, "min": 2, "max": 50, "step": 1, "tooltip": "Radius of active leading brush head"}),
                "optical_flow_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Intensity of secondary optical flow particle streamers"}),
                "particle_density": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Density of momentum particles"}),
                "glow_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05, "tooltip": "Soft luminescence bloom around active brush ribbons"}),
                "anchor_weight_wrists": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Weight for wrist gestures (primary dance marks)"}),
                "anchor_weight_feet": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Weight for ankle/feet floor movements"}),
                "anchor_weight_elbows_knees": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Weight for elbow and knee articulation"}),
                "anchor_weight_torso_head": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Weight for torso center of mass and head"}),
                "brush_color_mode": (["luminous_white", "kinetic_spectrum", "warm_amber", "cool_cyan"], {"default": "luminous_white", "tooltip": "Color palette preset"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "Output video frame rate"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("brush_stroke_frames", "brush_stroke_video_file")
    FUNCTION = "render_brush_strokes"
    CATEGORY = "KineticMotion"
    DESCRIPTION = "Converts motion trajectories and splines into abstract, progressive dynamic brushstrokes on canvas."

    def _catmull_rom(self, pts, num_samples=8):
        if len(pts) < 2: return pts
        if len(pts) == 2: return pts
        pts = [pts[0]] + list(pts) + [pts[-1]]
        curve = []
        for i in range(len(pts) - 3):
            p0 = np.array(pts[i], dtype=float)
            p1 = np.array(pts[i+1], dtype=float)
            p2 = np.array(pts[i+2], dtype=float)
            p3 = np.array(pts[i+3], dtype=float)
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3)
                curve.append((int(round(pt[0])), int(round(pt[1]))))
        curve.append(pts[-2])
        return curve

    def render_brush_strokes(
        self,
        motion_input,
        trajectory_history_length=24,
        temporal_decay=0.88,
        stroke_base_width=22,
        min_stroke_width=3,
        max_stroke_width=48,
        velocity_influence=1.8,
        acceleration_influence=0.8,
        brush_head_size=14,
        optical_flow_strength=0.4,
        particle_density=0.5,
        glow_strength=0.6,
        anchor_weight_wrists=2.2,
        anchor_weight_feet=1.5,
        anchor_weight_elbows_knees=1.0,
        anchor_weight_torso_head=0.8,
        brush_color_mode="luminous_white",
        fps=24
    ):
        # Unpack motion_input
        motion_data = None
        if isinstance(motion_input, dict) and "history_pts" in motion_input:
            motion_data = motion_input
        elif isinstance(motion_input, tuple) and len(motion_input) > 0 and isinstance(motion_input[-1], dict):
            motion_data = motion_input[-1]
        
        if motion_data is None:
            # Fallback: run extractor directly
            extractor = KineticMotionCurveExtractor()
            _, _, _, _, _, _, _, motion_data = extractor.extract_motion_representation(motion_input, fps=fps)

        history_pts = motion_data["history_pts"]
        history_vels = motion_data.get("history_vels", [])
        history_accels = motion_data.get("history_accels", [])
        flow_list = motion_data.get("flow_list", [])
        masks_list = motion_data.get("masks_list", [])
        fps = motion_data.get("fps", fps)
        w = motion_data["width"]
        h = motion_data["height"]
        num_frames = motion_data["num_frames"]

        # Motion Anchor Hierarchical Weighting
        ANCHOR_WEIGHTS = {
            15: anchor_weight_wrists,       # Left wrist
            16: anchor_weight_wrists,       # Right wrist
            27: anchor_weight_feet,         # Left ankle
            28: anchor_weight_feet,         # Right ankle
            13: anchor_weight_elbows_knees, # Left elbow
            14: anchor_weight_elbows_knees, # Right elbow
            25: anchor_weight_elbows_knees, # Left knee
            26: anchor_weight_elbows_knees, # Right knee
            0:  anchor_weight_torso_head,   # Head
            11: anchor_weight_torso_head * 0.7, # Left shoulder
            12: anchor_weight_torso_head * 0.7, # Right shoulder
            23: anchor_weight_torso_head,   # Left hip
            24: anchor_weight_torso_head    # Right hip
        }

        # Color Palette Profiles
        PALETTES = {
            "luminous_white": {15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.95, 1.0), 28: (0.95, 0.95, 1.0), "default": (0.9, 0.9, 0.9)},
            "kinetic_spectrum": {15: (1.0, 0.25, 0.55), 16: (0.0, 0.95, 1.0), 27: (0.65, 0.3, 1.0), 28: (1.0, 0.9, 0.1), "default": (0.7, 0.85, 1.0)},
            "warm_amber": {15: (1.0, 0.7, 0.1), 16: (1.0, 0.35, 0.1), 27: (1.0, 0.85, 0.4), 28: (1.0, 0.5, 0.2), "default": (0.9, 0.6, 0.2)},
            "cool_cyan": {15: (0.1, 0.95, 1.0), 16: (0.2, 0.6, 1.0), 27: (0.5, 0.4, 1.0), 28: (0.0, 0.8, 0.9), "default": (0.4, 0.7, 0.95)}
        }
        palette = PALETTES.get(brush_color_mode, PALETTES["luminous_white"])

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        rendered_frames = []
        flow_particles = []

        decay = np.clip(temporal_decay, 0.50, 0.99)
        max_particles = int(180 * particle_density)

        for t in range(num_frames):
            # 1. Temporal Persistence Canvas Decay
            canvas = canvas * decay
            stroke_layer = np.zeros((h, w, 3), dtype=np.float32)

            curr_pts = history_pts[t]
            mask = masks_list[t] if t < len(masks_list) else None
            flow = flow_list[t] if t < len(flow_list) else None

            # 2. Secondary Optical Flow Particle Streamlines
            if optical_flow_strength > 0 and flow is not None and curr_pts and max_particles > 0:
                for p_idx in [15, 16, 27, 28, 0, 13, 14]:
                    if p_idx in curr_pts:
                        px, py = curr_pts[p_idx]
                        if 0 <= px < w and 0 <= py < h:
                            if mask is None or (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]):
                                fx, fy = flow[py, px]
                                if np.hypot(fx, fy) > 1.2:
                                    color = palette.get(p_idx, palette["default"])
                                    flow_particles.append([float(px), float(py), float(fx), float(fy), 0, color])
                
                new_particles = []
                for p in flow_particles:
                    px, py, fx, fy, age, color = p
                    if age < 10 and 0 <= int(px) < w and 0 <= int(py) < h:
                        npx, npy = px + fx * 1.5, py + fy * 1.5
                        alpha = max(0.05, 1.0 - (age / 10.0)) * optical_flow_strength
                        c_p = (color[0] * alpha, color[1] * alpha, color[2] * alpha)
                        cv2.line(stroke_layer, (int(px), int(py)), (int(npx), int(npy)), c_p, int(max(1, 3 * alpha)), cv2.LINE_AA)
                        new_particles.append([npx, npy, fx * 0.88, fy * 0.88, age + 1, color])
                flow_particles = new_particles[-max_particles:]

            # 3. Progressive Spline Ribbon Strokes (strictly past window [t - history, t])
            window_start = max(0, t - trajectory_history_length)

            for p_idx, anchor_w in ANCHOR_WEIGHTS.items():
                if anchor_w <= 0.01: continue
                
                pts_seq = [history_pts[fi][p_idx] for fi in range(window_start, t + 1) if p_idx in history_pts[fi]]
                if len(pts_seq) >= 2:
                    spline_pts = self._catmull_rom(pts_seq, num_samples=6)
                    if len(spline_pts) >= 2:
                        speeds = []
                        for s_i in range(len(spline_pts) - 1):
                            dx = spline_pts[s_i+1][0] - spline_pts[s_i][0]
                            dy = spline_pts[s_i+1][1] - spline_pts[s_i][1]
                            speeds.append(np.hypot(dx, dy))
                        avg_spd = max(0.1, np.mean(speeds))

                        base_c = palette.get(p_idx, palette["default"])

                        # Draw Progressive Spline Ribbon
                        for s_i in range(len(spline_pts) - 1):
                            progress = (s_i + 1) / len(spline_pts) # 0 at oldest tail -> 1 at active brush head
                            local_spd = speeds[s_i]
                            spd_factor = np.clip(local_spd / (avg_spd + 1e-5), 0.4, 3.0) if velocity_influence > 0 else 1.0
                            
                            # Stroke width modulation
                            raw_w = progress * stroke_base_width * anchor_w * (1.0 + (velocity_influence - 1.0) * (spd_factor - 1.0) * 0.5)
                            w_clamped = int(np.clip(round(raw_w), min_stroke_width, max_stroke_width))
                            
                            # Luminescence & color gradient
                            lum = np.clip(progress * (0.6 + 0.4 * spd_factor), 0.1, 1.0)
                            seg_c = (base_c[0] * lum, base_c[1] * lum, base_c[2] * lum)
                            
                            cv2.line(stroke_layer, spline_pts[s_i], spline_pts[s_i+1], seg_c, w_clamped, cv2.LINE_AA)
                            cv2.circle(stroke_layer, spline_pts[s_i+1], w_clamped // 2, seg_c, -1, cv2.LINE_AA)

                        # Active Leading Brush Head (at current position P(t))
                        head_pt = spline_pts[-1]
                        head_spd = speeds[-1] if speeds else 1.0
                        head_spd_ratio = np.clip(head_spd / (avg_spd + 1e-5), 0.5, 3.0)
                        
                        head_r = int(np.clip(brush_head_size * anchor_w * (0.7 + 0.3 * head_spd_ratio), 3, max_stroke_width))
                        head_c = (base_c[0] * 1.0, base_c[1] * 1.0, base_c[2] * 1.0)
                        cv2.circle(stroke_layer, head_pt, head_r, head_c, -1, cv2.LINE_AA)

            # 4. Composite active strokes onto decaying canvas
            canvas = np.maximum(canvas, stroke_layer)

            # 5. Glow Bloom Post-Processing
            if glow_strength > 0:
                blurred = cv2.GaussianBlur(canvas, (21, 21), 0)
                frame_comp = np.clip(canvas + glow_strength * 0.5 * blurred, 0.0, 1.0)
            else:
                frame_comp = np.clip(canvas, 0.0, 1.0)

            rendered_frames.append((frame_comp * 255.0).astype(np.uint8))

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for rf in rendered_frames:
            if rf.shape[1] != w or rf.shape[0] != h:
                rf = cv2.resize(rf, (w, h))
            out_writer.write(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        out_tensor = torch.from_numpy(np.stack(rendered_frames, axis=0).astype(np.float32) / 255.0)
        print(f"[KineticMotionToBrushRenderer] Rendered {len(rendered_frames)} abstract brushstroke frames @ {fps}fps to {tmp_mp4_path}")
        return (out_tensor, tmp_mp4_path)


class KineticVideoCombine:
    """
    Combines image frame batches into MP4/WebP/GIF video with inline animated UI preview.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of image frames [B, H, W, C]"}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "Playback frame rate (FPS)"}),
                "format": (["mp4", "animated_webp", "gif"], {"default": "mp4"}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "Save output to ComfyUI output directory"}),
                "filename_prefix": ("STRING", {"default": "KineticVideo", "tooltip": "Prefix for saved video/animation files"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "KineticMotion"
    DESCRIPTION = "Combines image frames into a playable video/animation with inline preview."

    def combine_video(self, images, frame_rate=24, format="mp4", save_output=True, filename_prefix="KineticVideo"):
        if images is None or len(images.shape) < 4 or images.shape[0] == 0:
            return {"ui": {"images": []}, "result": (images, "")}

        output_dir = self.output_dir if save_output else self.temp_dir
        type_str = "output" if save_output else "temp"

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        h, w = images.shape[1], images.shape[2]
        num_frames = images.shape[0]

        pil_frames = []
        for idx in range(num_frames):
            frame_np = (images[idx].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_np))

        duration_ms = max(1, int(1000.0 / max(1, frame_rate)))
        saved_file_path = ""
        ui_results = []

        if format == "animated_webp":
            file_name = f"{filename}_{counter:05d}_.webp"
            saved_file_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(
                saved_file_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                quality=92,
                method=4
            )
            ui_results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/webp"
            })
        elif format == "gif":
            file_name = f"{filename}_{counter:05d}_.gif"
            saved_file_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(
                saved_file_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            ui_results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/gif"
            })
        else:  # mp4
            file_name = f"{filename}_{counter:05d}_.mp4"
            saved_file_path = os.path.join(full_output_folder, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(saved_file_path, fourcc, float(frame_rate), (w, h))
            for idx in range(num_frames):
                frame_bgr = cv2.cvtColor((images[idx].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()

            preview_webp_name = f"{filename}_{counter:05d}_preview.webp"
            preview_path = os.path.join(full_output_folder, preview_webp_name)
            pil_frames[0].save(
                preview_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                quality=85,
                method=3
            )
            ui_results.append({
                "filename": preview_webp_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/webp"
            })

        return {
            "ui": {"images": ui_results},
            "result": (images, saved_file_path)
        }


NODE_CLASS_MAPPINGS = {
    "KineticMotionCurveExtractor": KineticMotionCurveExtractor,
    "KineticMotionToBrushRenderer": KineticMotionToBrushRenderer,
    "KineticVideoCombine": KineticVideoCombine,
    "GeminiVideoCombine": KineticVideoCombine  # Backward compatibility alias
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KineticMotionCurveExtractor": "Google Kinetic Motion Curve Extractor",
    "KineticMotionToBrushRenderer": "Kinetic Motion-to-Brush Renderer",
    "KineticVideoCombine": "Kinetic Video Combine & Preview",
    "GeminiVideoCombine": "Gemini Video Combine & Preview"
}
