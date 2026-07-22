import math
import numpy as np
from collections import defaultdict

class CameraCalibrator:
    def __init__(self, calibration_window=100, default_car_width_m=1.8):
        """
        Gathers bounding box widths to estimate pixel-to-meter ratio.
        """
        self.calibration_window = calibration_window
        self.default_car_width_m = default_car_width_m
        self.car_widths_px = []
        self.calibrated = False
        self.meters_per_pixel = 1.8 / 50.0  # Default fallback (50px average width)

    def add_detections(self, bboxes, class_names, frame_number):
        """
        Collects widths of detected 'car' bounding boxes during the calibration window.
        """
        if self.calibrated or frame_number > self.calibration_window:
            return

        for bbox, cls_name in zip(bboxes, class_names):
            if cls_name == "car":
                # bbox format: [x1, y1, x2, y2]
                w_px = bbox[2] - bbox[0]
                if w_px > 5:  # filter out tiny boxes
                    self.car_widths_px.append(w_px)

    def calibrate(self):
        """
        Performs calibration based on gathered data.
        """
        if self.calibrated:
            return self.meters_per_pixel

        if len(self.car_widths_px) > 10:
            mean_width = np.mean(self.car_widths_px)
            if mean_width > 0:
                self.meters_per_pixel = self.default_car_width_m / mean_width
                print(f"[Calibration] Calibrated: mean car width = {mean_width:.1f} px | meters_per_pixel = {self.meters_per_pixel:.6f}")
        else:
            print(f"[Calibration] Warning: Insufficient car detections ({len(self.car_widths_px)}) for calibration. Using fallback.")
            
        self.calibrated = True
        return self.meters_per_pixel

class SpeedEstimator:
    def __init__(self, fps=30.0, k_window=10):
        """
        Computes speed based on displacement between c_t and c_{t-K}.
        """
        self.fps = fps
        self.k_window = k_window
        # self.history[track_id] = list of tuples (frame_number, cx, cy)
        self.history = defaultdict(list)
        self.last_speeds = defaultdict(float)

    def update_and_estimate(self, track_id, cx, cy, frame_number, meters_per_pixel):
        """
        Updates the track history and returns estimated speed in km/h.
        """
        track_hist = self.history[track_id]
        track_hist.append((frame_number, cx, cy))
        
        # Keep history clean (up to twice k_window to find matches easily)
        if len(track_hist) > self.k_window * 3:
            track_hist.pop(0)

        # Look for a historical point exactly k_window frames ago
        target_frame = frame_number - self.k_window
        
        prev_point = None
        for hist_frame, hx, hy in reversed(track_hist):
            if hist_frame <= target_frame:
                prev_point = (hist_frame, hx, hy)
                break

        if prev_point is not None:
            prev_frame, px, py = prev_point
            actual_k = frame_number - prev_frame
            if actual_k > 0:
                dx = cx - px
                dy = cy - py
                displacement_px = math.hypot(dx, dy)
                
                time_s = actual_k / self.fps
                speed_mps = (displacement_px * meters_per_pixel) / time_s
                speed_kmh = speed_mps * 3.6
                
                self.last_speeds[track_id] = speed_kmh
                return speed_kmh

        return self.last_speeds.get(track_id, 0.0)
