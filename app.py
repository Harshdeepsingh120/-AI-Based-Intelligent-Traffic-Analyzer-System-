import os
import json
import base64
import cv2
import time
import shutil
import threading
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from tracker import Sort
from speed_estimator import CameraCalibrator, SpeedEstimator
from signal_control import SignalController
from chatbot import Chatbot

app = FastAPI(title="AI Traffic Analyzer API")

# Ensure folders exist
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state
processing_state = {
    "status": "idle",
    "progress": 0.0,
    "current_frame": 0,
    "total_frames": 0,
    "left_density": 0,
    "right_density": 0,
    "car_count": 0,
    "bus_count": 0,
    "truck_count": 0,
    "person_count": 0,
    "motorcycle_count": 0,
    "avg_speed_left": 0.0,
    "avg_speed_right": 0.0,
    "signal_state": "GREEN_LEFT",
    "signal_timer": 10.0,
    "processed_count": 0,
    "error_message": "",
    "frame_b64": ""          # base64-encoded JPEG — no file I/O needed
}

# Cancel flag for background processing thread
cancel_processing = threading.Event()

# Instantiate global chatbot
chatbot_instance = Chatbot()

class ChatQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    """
    Serves the main frontend page.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

def _reset_and_start(video_path: str):
    """
    Shared helper: cancels any active processing thread, resets global state,
    and spawns a new background thread for the given video path.
    """
    global cancel_processing, processing_state

    cancel_processing.set()
    time.sleep(0.5)  # Let the old thread shut down gracefully
    cancel_processing.clear()

    processing_state = {
        "status": "processing",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "left_density": 0,
        "right_density": 0,
        "car_count": 0,
        "bus_count": 0,
        "truck_count": 0,
        "person_count": 0,
        "motorcycle_count": 0,
        "avg_speed_left": 0.0,
        "avg_speed_right": 0.0,
        "signal_state": "GREEN_LEFT",
        "signal_timer": 10.0,
        "processed_count": 0,
        "error_message": "",
        "frame_b64": ""
    }

    thread = threading.Thread(target=process_video_thread, args=(video_path,))
    thread.daemon = True
    thread.start()


def _reset_and_start_cached(cache_path: str):
    """
    Cancels any active processing thread, resets global state,
    and spawns a cached demo playback thread.
    """
    global cancel_processing, processing_state

    cancel_processing.set()
    time.sleep(0.3)
    cancel_processing.clear()

    processing_state = {
        "status": "processing",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "left_density": 0,
        "right_density": 0,
        "car_count": 0,
        "bus_count": 0,
        "truck_count": 0,
        "person_count": 0,
        "motorcycle_count": 0,
        "avg_speed_left": 0.0,
        "avg_speed_right": 0.0,
        "signal_state": "GREEN_LEFT",
        "signal_timer": 10.0,
        "processed_count": 0,
        "error_message": "",
        "frame_b64": ""
    }

    thread = threading.Thread(target=play_cached_demo_thread, args=(cache_path,))
    thread.daemon = True
    thread.start()


@app.post("/api/upload")
def upload_video(file: UploadFile = File(...)):
    """
    Receives an uploaded video file and spawns the background processing thread.
    """
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov"]:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, avi, mov.")

    # Save uploaded file
    video_path = os.path.join(UPLOAD_DIR, f"target_video{ext}")
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    _reset_and_start(video_path)
    return {"message": "Upload successful. Video analysis started.", "status": "processing"}


@app.post("/api/demo")
def start_demo():
    """
    Triggers analysis on the bundled demo clip. Uses pre-processed cache (static/demo_cache.json)
    if available for near-instant execution on CPU-constrained servers (e.g. Render free tier).
    """
    cache_path = os.path.join(STATIC_DIR, "demo_cache.json")
    demo_path = os.path.join(STATIC_DIR, "demo.mp4")

    if os.path.exists(cache_path):
        _reset_and_start_cached(cache_path)
        return {"message": "Cached demo analysis started.", "status": "processing"}
    elif os.path.exists(demo_path):
        _reset_and_start(demo_path)
        return {"message": "Demo analysis started.", "status": "processing"}
    else:
        raise HTTPException(status_code=404, detail="Demo video not found on server.")

@app.get("/api/status")
def get_status():
    """
    Returns current analysis progress and active metrics.
    """
    return processing_state

def play_cached_demo_thread(cache_path: str):
    """
    Background thread replaying pre-processed demo output from demo_cache.json.
    Paces playback frame-by-frame at ~16 FPS to update UI and chatbot state smoothly.
    """
    global processing_state, cancel_processing, chatbot_instance

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_frames = json.load(f)
    except Exception as e:
        processing_state["status"] = "failed"
        processing_state["error_message"] = f"Failed to read demo cache: {e}"
        return

    if not cached_frames:
        processing_state["status"] = "failed"
        processing_state["error_message"] = "Demo cache file is empty."
        return

    for frame_state in cached_frames:
        if cancel_processing.is_set():
            return

        processing_state.update(frame_state)

        chatbot_instance.update_state(
            left_count=frame_state.get("left_density", 0),
            right_count=frame_state.get("right_density", 0),
            signal_state=frame_state.get("signal_state", "GREEN_LEFT"),
            remaining_time=frame_state.get("signal_timer", 10.0),
            avg_speed_l=frame_state.get("avg_speed_left", 0.0),
            avg_speed_r=frame_state.get("avg_speed_right", 0.0),
            total_vehicles=frame_state.get("processed_count", 0)
        )

        time.sleep(0.06)

    if not cancel_processing.is_set():
        processing_state["status"] = "completed"
        processing_state["progress"] = 100.0


@app.post("/api/chat")
def chatbot_query(query_data: ChatQuery):
    """
    Interfaces with the hybrid chatbot module.
    """
    ans = chatbot_instance.query(query_data.query)
    return {"response": ans}

def process_video_thread(video_path):
    """
    Background thread running the pipeline on the target video.
    """
    global processing_state, cancel_processing, chatbot_instance
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        processing_state["status"] = "failed"
        processing_state["error_message"] = "Cannot open video file."
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    midpoint_x = width // 2

    processing_state["total_frames"] = total_frames
    
    # Initialize pipeline modules
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.50)
    calibrator = CameraCalibrator(calibration_window=100)
    speed_estimator = SpeedEstimator(fps=fps, k_window=10)
    signal_controller = SignalController(fps=fps, t_eval=5.0)

    # Class configs
    VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
    CLASS_COLORS = {
        "car": (0, 200, 0),
        "motorcycle": (255, 165, 0),
        "bus": (0, 100, 255),
        "truck": (0, 0, 200),
        "person": (0, 220, 220),
    }

    # Tracking sets
    total_vehicles_processed = set()
    frame_number = 0
    N_SKIP = 2  # Skip frames to speed up web processing

    # Class-specific counters
    class_totals = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "person": 0}
    seen_track_ids = set()

    try:
        while not cancel_processing.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            processing_state["current_frame"] = frame_number
            processing_state["progress"] = round((frame_number / total_frames) * 100, 1)

            tracked_objects = []

            # 1. Detection
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
                        
                        # Apply class-specific gates
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

                # Update Calibrator
                if frame_number <= 100:
                    calibrator.add_detections(
                        [t[0] for t in tracked_objects],
                        [t[2] for t in tracked_objects],
                        frame_number
                    )
                    if frame_number == 100:
                        calibrator.calibrate()
            else:
                # Interpolate skipped frames
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

            # Draw Midpoint
            cv2.line(annotated, (midpoint_x, 0), (midpoint_x, height), (255, 0, 0), 2)

            d_l = 0
            d_r = 0
            speeds_l = []
            speeds_r = []

            for bbox, tid, cls_name in tracked_objects:
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Class totals counter
                if tid not in seen_track_ids:
                    seen_track_ids.add(tid)
                    if cls_name in class_totals:
                        class_totals[cls_name] += 1

                # Speed
                speed_val = speed_estimator.update_and_estimate(tid, cx, cy, frame_number, meters_per_pixel)

                # Partitioning
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

                # Overlay box
                color = CLASS_COLORS.get(cls_name, (200, 200, 200))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{cls_name} #{tid} | {speed_val:.0f} km/h"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Signal Controller
            signal_controller.update(d_l, d_r, frame_number)
            sig_status = signal_controller.get_status()

            # Average speed
            avg_speed_l = np.mean(speeds_l) if speeds_l else 0.0
            avg_speed_r = np.mean(speeds_r) if speeds_r else 0.0

            # Update shared chatbot state
            chatbot_instance.update_state(
                left_count=d_l,
                right_count=d_r,
                signal_state=sig_status["state"],
                remaining_time=sig_status["timer"],
                avg_speed_l=avg_speed_l,
                avg_speed_r=avg_speed_r,
                total_vehicles=len(total_vehicles_processed)
            )

            # Update global state for API response
            processing_state["left_density"] = d_l
            processing_state["right_density"] = d_r
            processing_state["avg_speed_left"] = round(avg_speed_l, 1)
            processing_state["avg_speed_right"] = round(avg_speed_r, 1)
            processing_state["signal_state"] = sig_status["state"]
            processing_state["signal_timer"] = round(sig_status["timer"], 1)
            processing_state["processed_count"] = len(total_vehicles_processed)
            
            processing_state["car_count"] = class_totals["car"]
            processing_state["bus_count"] = class_totals["bus"]
            processing_state["truck_count"] = class_totals["truck"]
            processing_state["person_count"] = class_totals["person"]
            processing_state["motorcycle_count"] = class_totals["motorcycle"]

            # Encode the annotated frame as a base64 JPEG string and store
            # it directly in the shared state dict. The frontend reads it from
            # /api/status and sets img.src to a data URL. This completely
            # eliminates file I/O and the Windows file-lock race condition.
            ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                processing_state["frame_b64"] = base64.b64encode(buf).decode('utf-8')

    except Exception as e:
        processing_state["status"] = "failed"
        processing_state["error_message"] = str(e)
        cap.release()
        return

    cap.release()
    if not cancel_processing.is_set():
        processing_state["status"] = "completed"
        processing_state["progress"] = 100.0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
