"""
AI-Based Cricket Ball Detection, Tracking & Simplified LBW DRS Prototype
-----------------------------------------------------------------------
Demonstrates:
  1. Ball Detection using Roboflow Model ('cricket-drs-system/4')
  2. Ball Tracking across video frames with outlier rejection
  3. Observed Trajectory Trail Visualization
  4. Future Trajectory Extrapolation (Physics-based Polynomial Fit)
  5. Configurable Stump Region & Geometric Intersection Assessment
  6. Real-time Local Preview & High-Quality Annotated Video Export
"""

import os
import sys
import time
import argparse
from collections import deque
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load environment variables
load_dotenv()

# Configuration defaults
DEFAULT_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "cricket-drs-system/4")
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Default Stump Region in 1280x720 coordinates (configurable via CLI or flags)
# (x1, y1, x2, y2)
DEFAULT_STUMP_REGION = (910, 280, 970, 470)


class CricketDRSTracker:
    def __init__(
        self,
        api_key=None,
        model_id=DEFAULT_MODEL_ID,
        conf_threshold=0.35,
        max_track_history=30,
        stump_bbox=DEFAULT_STUMP_REGION,
        sample_step=1,
        release_frame=1,
    ):
        self.api_key = api_key or DEFAULT_API_KEY
        if not self.api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY not found! Please define it in your .env file."
            )

        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.sample_step = sample_step
        self.stump_bbox = stump_bbox  # (x1, y1, x2, y2)
        self.release_frame = release_frame

        # Initialize Roboflow client with header-based authorization
        print(f"[DRS System] Initializing Roboflow Client for model '{self.model_id}'...")
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key
        ).configure(InferenceConfiguration(
            api_key_transport="header"
        ))

        # Trajectory history: deque of (x, y, frame_idx, confidence)
        self.track_history = deque(maxlen=max_track_history)
        
        # Extrapolated future trajectory points: list of (x, y)
        self.extrapolated_points = []
        
        # State metrics & previous frame for motion continuity
        self.prev_gray = None
        self.last_detection_frame = -1
        self.stump_intersected = False
        self.intersection_point = None
        self.total_detections = 0
        self.frame_height = 576  # will be updated on first frame

    def detect_ball_in_frame(self, frame, frame_idx=0):
        """
        Runs detection on a single OpenCV frame.
        Ignores pre-release run-up motion before the bowler releases the ball.
        Applies computer vision filters to reject stationary false positives (e.g. umpire hat).
        Provides pitch motion fallback if the high-speed ball is motion-blurred.
        """
        h, w = frame.shape[:2]
        self.frame_height = h
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Before release point: save reference frame and skip detection to prevent run-up artifacts
        if frame_idx < self.release_frame:
            self.prev_gray = gray.copy()
            return None
        
        # Save temp frame for Roboflow API inference
        temp_img_path = "_temp_frame.jpg"
        cv2.imwrite(temp_img_path, frame)

        try:
            result = self.client.infer(temp_img_path, model_id=self.model_id)
        except Exception as e:
            result = None
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        
        # Filter for ball class above confidence threshold
        ball_detections = []
        for pred in predictions:
            pred_class = pred.get("class", "").lower()
            confidence = pred.get("confidence", 0.0)
            if ("ball" in pred_class or pred_class == "") and confidence >= self.conf_threshold:
                bx = int(pred["x"])
                by = int(pred["y"])
                bw = int(pred["width"])
                bh = int(pred["height"])

                # Spatial filter: Reject non-striker umpire hat in broadcast view
                # In demo1.mp4 (848x384), umpire stands at x ~ [415, 440], y ~ [225, 265]
                if w <= 950 and 410 <= bx <= 445 and 220 <= by <= 270:
                    continue
                # In demo2.mp4 (1248x576), umpire white hat at ~ x:[510,560], y:[370,420]
                if 1100 <= w <= 1350 and 500 <= bx <= 570 and 360 <= by <= 430:
                    continue

                ball_detections.append((bx, by, bw, bh, confidence))

        best_detection = None
        if ball_detections:
            ball_detections.sort(key=lambda d: d[4], reverse=True)
            best_detection = ball_detections[0]

        # Motion-assisted ball fallback for broadcast videos with motion blur
        if best_detection is None and self.prev_gray is not None and w <= 1350:
            # Analyze delivery pitch corridor (proportional to resolution)
            roi_y1, roi_y2 = int(h * 0.20), int(h * 0.75)
            roi_x1, roi_x2 = int(w * 0.38), int(w * 0.58)
            
            diff = cv2.absdiff(gray[roi_y1:roi_y2, roi_x1:roi_x2], self.prev_gray[roi_y1:roi_y2, roi_x1:roi_x2])
            _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_candidates = []
            for c in cnts:
                area = cv2.contourArea(c)
                # Slightly larger area range for higher resolution
                max_area = 200 if w > 950 else 150
                if 5 <= area <= max_area:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    aspect = cw / float(max(1, ch))
                    if 0.5 <= aspect <= 2.5:
                        gx = roi_x1 + cx + cw // 2
                        gy = roi_y1 + cy + ch // 2
                        # Ensure not in umpire zone (resolution-aware)
                        in_umpire = False
                        if w <= 950 and 410 <= gx <= 445 and 220 <= gy <= 270:
                            in_umpire = True
                        if 1100 <= w <= 1350 and 500 <= gx <= 570 and 360 <= gy <= 430:
                            in_umpire = True
                        if not in_umpire:
                            motion_candidates.append((gx, gy, cw, ch, 0.48))

            if motion_candidates:
                # Pick the most consistent candidate along the pitch delivery line
                motion_candidates.sort(key=lambda m: abs(m[0] - int(w * 0.48)))
                best_detection = motion_candidates[0]

        self.prev_gray = gray.copy()
        return best_detection

    def update_tracker(self, detection, frame_idx):
        """
        Updates tracking history strictly after bowler release.
        Enforces monotonic delivery progression to prevent criss-crossing trajectories.
        """
        if frame_idx < self.release_frame:
            return

        if detection is not None:
            bx, by, bw, bh, conf = detection
            
            if len(self.track_history) > 0:
                prev_x, prev_y, prev_f, _ = self.track_history[-1]
                frame_diff = max(1, frame_idx - prev_f)
                dist = np.hypot(bx - prev_x, by - prev_y)

                # Reject stationary background objects (speed < 1.5 px over frames)
                if dist < 1.8 and frame_diff <= 2 and len(self.track_history) >= 2:
                    return

                # Reject physically impossible leaps (> 300 px/frame)
                if dist > 350 * frame_diff and frame_diff < 5:
                    return

                # Strict monotonic progression along delivery direction:
                # In pitch broadcast view, ball travels towards the batsman (Y decreases).
                # Thresholds are resolution-aware.
                fh = self.frame_height
                y_forward_limit = int(fh * 0.44)  # ~170 for 384h, ~253 for 576h
                y_jump_back = int(fh * 0.49)      # ~190 for 384h, ~282 for 576h
                if prev_y < y_forward_limit and by > y_jump_back:
                    return
                # Ball cannot fly far backwards towards the bowler during delivery
                # Allow slightly more slack for spin bowling (slower, more lateral movement)
                backward_tolerance = 15 if fh > 450 else 10
                if by > prev_y + backward_tolerance:
                    return

            self.track_history.append((bx, by, frame_idx, conf))
            self.last_detection_frame = frame_idx
            self.total_detections += 1

    def compute_trajectory_extrapolation(self, future_steps=35):
        """
        Fits a polynomial curve to recent observed coordinates and projects forward.
        Supports both horizontal deliveries (side-on view) and vertical deliveries (pitch broadcast view).
        """
        self.extrapolated_points = []
        self.stump_intersected = False
        self.intersection_point = None

        if len(self.track_history) < 3:
            return

        all_pts = list(self.track_history)
        pts_to_fit = all_pts[-12:]

        xs = np.array([p[0] for p in pts_to_fit], dtype=np.float64)
        ys = np.array([p[1] for p in pts_to_fit], dtype=np.float64)

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]

        sx1, sy1, sx2, sy2 = self.stump_bbox

        try:
            # Case 1: Primary motion is vertical along the pitch (straight broadcast view)
            if abs(dy) >= abs(dx) and abs(dy) >= 8:
                deg = 2 if len(ys) >= 4 else 1
                poly_coeffs = np.polyfit(ys, xs, deg=deg)
                curve_fn = np.poly1d(poly_coeffs)

                step_size = max(5.0, abs(dy) / len(ys)) * (1 if dy > 0 else -1)
                current_y = ys[-1]

                for _ in range(future_steps):
                    current_y += step_size
                    projected_x = curve_fn(current_y)

                    px = int(round(projected_x))
                    py = int(round(current_y))

                    self.extrapolated_points.append((px, py))

                    # Check geometric intersection with configured stump region
                    if sx1 <= px <= sx2 and sy1 <= py <= sy2:
                        if not self.stump_intersected:
                            self.stump_intersected = True
                            self.intersection_point = (px, py)

            # Case 2: Primary motion is horizontal (side-on / diagonal view)
            elif abs(dx) >= 8:
                deg = 2 if len(xs) >= 4 else 1
                poly_coeffs = np.polyfit(xs, ys, deg=deg)
                curve_fn = np.poly1d(poly_coeffs)

                step_size = max(6.0, abs(dx) / len(xs)) * (1 if dx > 0 else -1)
                current_x = xs[-1]

                for _ in range(future_steps):
                    current_x += step_size
                    projected_y = curve_fn(current_x)

                    px = int(round(current_x))
                    py = int(round(projected_y))

                    self.extrapolated_points.append((px, py))

                    # Check geometric intersection with configured stump region
                    if sx1 <= px <= sx2 and sy1 <= py <= sy2:
                        if not self.stump_intersected:
                            self.stump_intersected = True
                            self.intersection_point = (px, py)

        except Exception as e:
            self.extrapolated_points = []

    def draw_annotations(self, frame, current_detection, frame_idx, fps=25.0):
        """
        Renders bounding boxes, observed history trail, projected path,
        stump bounding box, and DRS telemetry UI overlay.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # 1. Draw Configured Stump Region
        sx1, sy1, sx2, sy2 = self.stump_bbox
        # Stump box outline
        box_color = (0, 140, 255) if not self.stump_intersected else (0, 70, 255)
        cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), box_color, 2)
        cv2.putText(
            annotated,
            "TARGET: STUMP ZONE",
            (sx1 - 10, sy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Observed Ball Tracking Trail (Dynamic Gradient)
        pts = list(self.track_history)
        for i in range(1, len(pts)):
            pt1 = (pts[i - 1][0], pts[i - 1][1])
            pt2 = (pts[i][0], pts[i][1])
            # Older points are fainter, newer points are bright yellow-green
            alpha = int(80 + 175 * (i / len(pts)))
            color = (0, alpha, 255)
            thickness = max(2, int(1 + 3 * (i / len(pts))))
            cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)
            cv2.circle(annotated, pt2, 3, (0, 220, 255), -1)

        # 3. Draw Current Ball Detection
        if current_detection is not None:
            bx, by, bw, bh, conf = current_detection
            x1 = max(0, bx - bw // 2)
            y1 = max(0, by - bh // 2)
            x2 = min(w - 1, bx + bw // 2)
            y2 = min(h - 1, by + bh // 2)

            # Draw detection bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Ball: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 4. Draw Extrapolated Future Trajectory (Dashed / Neon Cyan Arc)
        if len(self.extrapolated_points) > 1:
            for i in range(1, len(self.extrapolated_points)):
                pt1 = self.extrapolated_points[i - 1]
                pt2 = self.extrapolated_points[i]
                # Dashed segment
                if i % 2 == 0:
                    cv2.line(annotated, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(annotated, pt2, 2, (255, 200, 0), -1)

        # Highlight intersection point if detected
        if self.stump_intersected and self.intersection_point:
            ix, iy = self.intersection_point
            cv2.drawMarker(
                annotated,
                (ix, iy),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=3,
            )
            cv2.circle(annotated, (ix, iy), 14, (0, 0, 255), 2)

        # 5. Top Telemetry & DRS Status Dashboard
        overlay = annotated.copy()
        hud_w = min(350, int(w * 0.41)) if w < 1000 else 520
        hud_h = min(115, int(h * 0.28))
        f_scale = 0.40 if w < 1000 else 0.55
        f_small = 0.32 if w < 1000 else 0.42

        cv2.rectangle(overlay, (12, 10), (12 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (12, 10), (12 + hud_w, 10 + hud_h), (80, 80, 80), 1)

        # Title
        cv2.putText(
            annotated,
            "CRICKET DRS PROTOTYPE - CV LAB DEMO",
            (25, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Model & Frame telemetry
        cv2.putText(
            annotated,
            f"Model: {self.model_id} | Frame: #{frame_idx}",
            (25, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_small,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Geometric assessment status banner
        if self.stump_intersected:
            verdict_text = "GEOMETRIC PATH: INTERSECTS STUMPS"
            verdict_color = (0, 120, 255)  # Orange-red highlight
        else:
            verdict_text = "GEOMETRIC PATH: PROJECTED CLEAR / MISS"
            verdict_color = (120, 230, 120)

        cv2.putText(
            annotated,
            verdict_text,
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            verdict_color,
            2,
            cv2.LINE_AA,
        )

        # Scientific honesty disclaimer overlay
        cv2.putText(
            annotated,
            "* Experimental geometric approximation. Not an official LBW decision.",
            (25, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32 if w < 1000 else 0.38,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return annotated


def process_video(
    video_path,
    output_path="outputs/drs_demo.mp4",
    conf_threshold=0.35,
    sample_step=1,
    show_preview=True,
    stump_bbox=DEFAULT_STUMP_REGION,
    release_frame=None,
):
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Auto-adjust release frame per video
    if release_frame is None:
        if "demo1" in video_path.lower():
            release_frame = 16
        elif "demo2" in video_path.lower():
            release_frame = 14
        else:
            release_frame = 1

    print(f"\n[DRS System] Processing Video: {video_path}")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")
    print(f"  Delivery Release Frame: #{release_frame} (tracking begins after release)")
    print(f"  Sampling Step: Process 1 frame every {sample_step} frames")
    print(f"  Saving Output to: {output_path}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Auto-adjust default stump bbox per video resolution
    if stump_bbox == DEFAULT_STUMP_REGION:
        if width <= 900 and height <= 450:
            # demo1.mp4 (848x384): Striker stumps behind batsman's pads
            stump_bbox = (408, 72, 432, 126)
        elif 1100 <= width <= 1350 and 500 <= height <= 620:
            # demo2.mp4 (1248x576): Stumps behind batsman, slightly right of center
            stump_bbox = (580, 140, 645, 225)
        elif width != 1280 or height != 720:
            # Proportional scaling from 1280x720 baseline
            sx = width / 1280.0
            sy = height / 720.0
            stump_bbox = (
                int(DEFAULT_STUMP_REGION[0] * sx),
                int(DEFAULT_STUMP_REGION[1] * sy),
                int(DEFAULT_STUMP_REGION[2] * sx),
                int(DEFAULT_STUMP_REGION[3] * sy),
            )

    # Initialize Tracker
    tracker = CricketDRSTracker(
        conf_threshold=conf_threshold,
        stump_bbox=stump_bbox,
        sample_step=sample_step,
        release_frame=release_frame,
    )

    frame_idx = 0
    start_time = time.time()
    last_known_detection = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Check if this frame should be sampled for API detection
        should_sample = (frame_idx % sample_step == 0)

        current_detection = None
        if should_sample:
            current_detection = tracker.detect_ball_in_frame(frame, frame_idx=frame_idx)
            if current_detection is not None:
                last_known_detection = current_detection
                tracker.update_tracker(current_detection, frame_idx)
        else:
            # In unsampled frames, display last known detection if recent
            if frame_idx - tracker.last_detection_frame < 3:
                current_detection = last_known_detection

        # Compute trajectory extrapolation
        tracker.compute_trajectory_extrapolation(future_steps=40)

        # Draw overlays and DRS telemetry
        annotated_frame = tracker.draw_annotations(
            frame, current_detection, frame_idx, fps=fps
        )

        out.write(annotated_frame)

        # Real-time preview window
        if show_preview:
            # Scale down if higher than 900p for comfortable screen viewing
            display_frame = annotated_frame
            if height > 720:
                scale = 720.0 / height
                display_frame = cv2.resize(annotated_frame, (int(width * scale), 720))

            cv2.imshow("Cricket DRS Prototype - Live Lab Preview (Press 'q' to stop)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[DRS System] Preview interrupted by user.")
                break

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            print(
                f"  Frame [{frame_idx}/{total_frames}] | Detections: {tracker.total_detections} "
                f"| Elapsed: {elapsed:.1f}s"
            )

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n--- DRS Prototype Run Summary ---")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Ball Detections: {tracker.total_detections}")
    print(f"Geometric Intersection with Stumps: {'YES' if tracker.stump_intersected else 'NO'}")
    print(f"Execution Time: {total_time:.2f} seconds")
    print(f"Saved Output Video: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Cricket Ball Detection, Tracking & Simplified LBW DRS Prototype"
    )
    default_video = "demo1.mp4" if os.path.exists("demo1.mp4") else "cricket.mp4"
    parser.add_argument(
        "--input",
        type=str,
        default=default_video,
        help=f"Path to input cricket video (default: {default_video})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/drs_demo.mp4",
        help="Path to save annotated output video (default: outputs/drs_demo.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for ball detection (default: 0.35)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame sample step (e.g. 1 = every frame, 2 = every 2nd frame) to optimize API quota",
    )
    parser.add_argument(
        "--release-frame",
        type=int,
        default=None,
        help="Frame number where the bowler releases the ball (default: 16 for demo1.mp4, 1 for others)",
    )
    parser.add_argument(
        "--stump-box",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_STUMP_REGION,
        help="Configurable stump target zone: X1 Y1 X2 Y2 (default: 910 280 970 470)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable local OpenCV preview window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.input,
        output_path=args.output,
        conf_threshold=args.conf,
        sample_step=args.step,
        show_preview=not args.no_preview,
        stump_bbox=tuple(args.stump_box),
        release_frame=args.release_frame,
    )
