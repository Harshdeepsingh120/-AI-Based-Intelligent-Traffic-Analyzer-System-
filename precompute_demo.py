import os
import json
import base64
import gc
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Sort
from speed_estimator import CameraCalibrator, SpeedEstimator
from signal_control import SignalController

try:
    import torch
except ImportError:
    torch = None

def precompute_demo_cache(video_path="static/demo.mp4", output_cache_path="static/demo_cache.json", batch_size=50):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print(f"Starting pre-processing for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    midpoint_x = width // 2

    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.50)
    calibrator = CameraCalibrator(calibration_window=100)
    speed_estimator = SpeedEstimator(fps=fps, k_window=10)
    signal_controller = SignalController(fps=fps, t_eval=5.0)

    VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
    CLASS_COLORS = {
        "car": (0, 200, 0),
        "motorcycle": (255, 165, 0),
        "bus": (0, 100, 255),
        "truck": (0, 0, 200),
        "person": (0, 220, 220),
    }

    total_vehicles_processed = set()
    frame_number = 0
    N_SKIP = 1  # Process every frame for smooth motion

    class_totals = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "person": 0}
    seen_track_ids = set()

    cached_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        tracked_objects = []

        if frame_number % N_SKIP == 0 or frame_number == 1:
            results = model.predict(frame, conf=0.25, iou=0.50, verbose=False)
            dets = []
            cls_names = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    conf_thresh = 0.45
                    if label == "person":
                        conf_thresh = 0.55
                    elif label in ["bus", "truck", "motorcycle"]:
                        conf_thresh = 0.25
                        
                    if label in ["car", "motorcycle", "bus", "truck", "person"] and conf >= conf_thresh:
                        dets.append([x1, y1, x2, y2, conf])
                        cls_names.append(label)

            dets_arr = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
            tracked_objects = tracker.update(dets_arr, cls_names)

            if frame_number <= 100:
                calibrator.add_detections(
                    [t[0] for t in tracked_objects],
                    [t[2] for t in tracked_objects],
                    frame_number
                )
                if frame_number == 100:
                    calibrator.calibrate()
            del results
        else:
            to_del = []
            for t, trk in enumerate(tracker.trackers):
                pos = trk.predict()
                if np.any(np.isnan(pos)):
                    to_del.append(t)
                elif trk.hits >= tracker.min_hits and trk.time_since_update <= tracker.max_age:
                    tracked_objects.append((pos, trk.id, trk.cls_name))
            tracker.trackers = [tracker.trackers[i] for i in range(len(tracker.trackers)) if i not in to_del]

        meters_per_pixel = calibrator.meters_per_pixel
        annotated = frame.copy()

        cv2.line(annotated, (midpoint_x, 0), (midpoint_x, height), (255, 0, 0), 2)

        d_l = 0
        d_r = 0
        speeds_l = []
        speeds_r = []

        for bbox, tid, cls_name in tracked_objects:
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if tid not in seen_track_ids:
                seen_track_ids.add(tid)
                if cls_name in class_totals:
                    class_totals[cls_name] += 1

            speed_val = speed_estimator.update_and_estimate(tid, cx, cy, frame_number, meters_per_pixel)

            if cx < midpoint_x:
                if cls_name in VEHICLE_CLASSES:
                    d_l += 1
                    speeds_l.append(speed_val)
            else:
                if cls_name in VEHICLE_CLASSES:
                    d_r += 1
                    speeds_r.append(speed_val)

            if cls_name in VEHICLE_CLASSES:
                total_vehicles_processed.add(tid)

            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"{cls_name} #{tid} | {speed_val:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        signal_controller.update(d_l, d_r, frame_number)
        sig_status = signal_controller.get_status()

        avg_speed_l = float(np.mean(speeds_l)) if speeds_l else 0.0
        avg_speed_r = float(np.mean(speeds_r)) if speeds_r else 0.0

        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
        frame_b64 = base64.b64encode(buf).decode('utf-8') if ok else ""

        progress = round((frame_number / total_frames) * 100, 1) if total_frames > 0 else 0

        state_snapshot = {
            "status": "processing",
            "progress": progress,
            "current_frame": frame_number,
            "total_frames": total_frames,
            "left_density": d_l,
            "right_density": d_r,
            "car_count": class_totals["car"],
            "bus_count": class_totals["bus"],
            "truck_count": class_totals["truck"],
            "person_count": class_totals["person"],
            "motorcycle_count": class_totals["motorcycle"],
            "avg_speed_left": round(avg_speed_l, 1),
            "avg_speed_right": round(avg_speed_r, 1),
            "signal_state": sig_status["state"],
            "signal_timer": round(sig_status["timer"], 1),
            "processed_count": len(total_vehicles_processed),
            "error_message": "",
            "frame_b64": frame_b64
        }
        cached_frames.append(state_snapshot)

        # Batch save to disk and garbage collect
        if frame_number % batch_size == 0 or frame_number == total_frames:
            temp_cache_path = output_cache_path + ".tmp"
            with open(temp_cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_frames, f)
            os.replace(temp_cache_path, output_cache_path)
            print(f"Incremental batch saved: {frame_number}/{total_frames} frames ({progress}%) written to {output_cache_path}.", flush=True)
            
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    cap.release()
    print(f"Pre-processing complete. Total {len(cached_frames)} cached frames saved to {output_cache_path}.", flush=True)

if __name__ == "__main__":
    precompute_demo_cache()

