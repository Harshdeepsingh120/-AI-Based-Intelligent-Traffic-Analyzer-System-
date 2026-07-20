import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tracker import Sort
from speed_estimator import CameraCalibrator, SpeedEstimator
from signal_control import SignalController
from chatbot import Chatbot

def select_video_scenario():
    """
    Prompts the user to select one of the available video files.
    """
    videos = [f for f in os.listdir(".") if f.endswith(".mp4")]
    if not videos:
        print("ERROR: No .mp4 videos found in the current directory.")
        return None
    
    print("\n--- Available Video Scenarios ---")
    for idx, vid in enumerate(videos):
        print(f"[{idx + 1}] {vid}")
    
    # Non-interactive fallback if inputs are piped, otherwise prompt
    try:
        choice = input(f"Select a scenario (1-{len(videos)}) [default: 1]: ").strip()
        if not choice:
            return videos[0]
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(videos):
            return videos[choice_idx]
    except Exception:
        pass
    
    print(f"Using default: {videos[0]}")
    return videos[0]

def main():
    # 1. SELECT VIDEO SCENARIO
    video_file = select_video_scenario()
    if not video_file:
        return

    # 2. CONFIGURATION
    MODEL_NAME = "yolov8n.pt"
    VEHICLE_CONF = 0.45
    PEDESTRIAN_CONF = 0.60
    IOU_THRESHOLD = 0.50
    N_SKIP = 2  # Configurable frame-skip: process every 2nd frame, interpolate on others
    
    VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
    PEDESTRIAN_CLASS = "person"
    
    CLASS_COLORS = {
        "car": (0, 200, 0),        # Green
        "motorcycle": (255, 165, 0), # Orange
        "bus": (0, 100, 255),      # Orange-red
        "truck": (0, 0, 200),      # Red
        "person": (0, 220, 220),   # Yellow-cyan
    }

    # 3. INITIALIZE MODELS & COMPONENTS
    print(f"\n[System] Loading model {MODEL_NAME}...")
    if not os.path.exists(MODEL_NAME):
        print(f"[System] Downloading {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
        print("[System] Model loaded successfully.")
    except Exception as e:
        print("[System] Model loading failed:", e)
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    midpoint_x = width // 2
    print(f"[System] Video {video_file} loaded | Resolution: {width}x{height} | FPS: {fps:.1f}")

    # Modules
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=IOU_THRESHOLD)
    calibrator = CameraCalibrator(calibration_window=100)
    speed_estimator = SpeedEstimator(fps=fps, k_window=10)
    signal_controller = SignalController(fps=fps, t_eval=5.0, t_min=10.0, t_max=60.0)
    chatbot = Chatbot()

    # Shared running states
    frame_number = 0
    total_vehicles_processed = set()
    active_speeds = {}

    print("\nStarting processing... Press 'q' in the window to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # We will keep a dictionary of the tracked objects on this frame
        current_frame_tracks = []

        # Run detection and update tracker only on non-skipped frames
        if frame_number % N_SKIP == 0 or frame_number == 1:
            # Predict and update with detections
            results = model.predict(
                frame,
                conf=min(VEHICLE_CONF, PEDESTRIAN_CONF), # run with lowest to get both
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
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
                    
                    # Apply class-specific confidence gates
                    conf_thresh = 0.45
                    if label == "person":
                        conf_thresh = 0.55  # Keep higher to prevent false detections of signs/poles
                    elif label in ["bus", "truck", "motorcycle"]:
                        conf_thresh = 0.25  # Lower to recover recall on rarer/occluded classes
                        
                    if label in ["car", "motorcycle", "bus", "truck", "person"] and conf >= conf_thresh:
                        dets.append([x1, y1, x2, y2, conf])
                        cls_names.append(label)

            dets_arr = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
            
            # Update Sort Tracker
            tracked_objects = tracker.update(dets_arr, cls_names)
            
            # Calibration segment (gather data for first 100 frames)
            if frame_number <= 100:
                calibrator.add_detections(
                    [t[0] for t in tracked_objects],
                    [t[2] for t in tracked_objects],
                    frame_number
                )
                if frame_number == 100:
                    calibrator.calibrate()

        else:
            # SKIPPED FRAME: Interpolate track positions by calling predict on active trackers
            tracked_objects = []
            to_del = []
            for t, trk in enumerate(tracker.trackers):
                pos = trk.predict()
                if np.any(np.isnan(pos)):
                    to_del.append(t)
                elif trk.hits >= tracker.min_hits and trk.time_since_update <= tracker.max_age:
                    tracked_objects.append((pos, trk.id, trk.cls_name))
            
            tracker.trackers = [tracker.trackers[i] for i in range(len(tracker.trackers)) if i not in to_del]

        # Use current calibration factor
        meters_per_pixel = calibrator.meters_per_pixel

        annotated = frame.copy()
        
        # Draw midpoint lane partition line
        cv2.line(annotated, (midpoint_x, 0), (midpoint_x, height), (255, 0, 0), 2)
        cv2.putText(annotated, "LEFT LANE", (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(annotated, "RIGHT LANE", (width - 250, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Analyze tracked objects
        d_l = 0
        d_r = 0
        speeds_l = []
        speeds_r = []

        for bbox, tid, cls_name in tracked_objects:
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Estimate speed
            speed_val = speed_estimator.update_and_estimate(tid, cx, cy, frame_number, meters_per_pixel)
            active_speeds[tid] = speed_val

            # Lane partitioning
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

            # Draw track bounding box and ID / speed overlay
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            tag = f"{cls_name} ID:{tid} | {speed_val:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)

        # Update signal control state machine
        signal_controller.update(d_l, d_r, frame_number)
        sig_status = signal_controller.get_status()

        # Update chatbot state
        avg_speed_l = np.mean(speeds_l) if speeds_l else 0.0
        avg_speed_r = np.mean(speeds_r) if speeds_r else 0.0
        chatbot.update_state(
            left_count=d_l,
            right_count=d_r,
            signal_state=sig_status["state"],
            remaining_time=sig_status["timer"],
            avg_speed_l=avg_speed_l,
            avg_speed_r=avg_speed_r,
            total_vehicles=len(total_vehicles_processed)
        )

        # Draw UI overlay panel at the top
        # Create semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # Draw signal control overlays
        sig_state = sig_status["state"]
        sig_timer = sig_status["timer"]
        
        if sig_state == "GREEN_LEFT":
            sig_color = (0, 255, 0)
            sig_text = f"SIGNAL: GREEN LEFT ({sig_timer:.1f}s)"
        elif sig_state == "GREEN_RIGHT":
            sig_color = (0, 255, 0)
            sig_text = f"SIGNAL: GREEN RIGHT ({sig_timer:.1f}s)"
        else:
            sig_color = (0, 0, 255)
            sig_text = f"SIGNAL: ALL RED ({sig_timer:.1f}s)"

        cv2.putText(annotated, sig_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sig_color, 2)
        
        # Display counts and speeds
        cv2.putText(annotated, f"Left Lane: {d_l} vehicles (avg {avg_speed_l:.0f} km/h)", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Right Lane: {d_r} vehicles (avg {avg_speed_r:.0f} km/h)", 
                    (midpoint_x + 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Overall processed count
        cv2.putText(annotated, f"Processed: {len(total_vehicles_processed)} vehicles", 
                    (width - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Traffic Analyzer", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[System] Execution stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[System] Video processing completed.")

    # 4. INTERACTIVE CHATBOT INTERFACE (POST-RUN)
    print("\n" + "="*50)
    print("=== TRAFFIC ANALYZER CHATBOT INTERFACE ===")
    print("="*50)
    print("The system run has finished. You can now query the chatbot.")
    print("Available intents: vehicle count, signal status, average speeds, lane density, general status.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query_str = input("Query: ").strip()
            if not query_str:
                continue
            if query_str.lower() in ["exit", "quit"]:
                print("Ending chatbot session. Goodbye!")
                break
            
            response = chatbot.query(query_str)
            print(f"Chatbot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEnding chatbot session. Goodbye!")
            break

if __name__ == "__main__":
    main()
