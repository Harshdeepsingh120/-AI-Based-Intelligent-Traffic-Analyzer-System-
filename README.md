# -AI-Based-Intelligent-Traffic-Analyzer-System-
Real-time AI traffic analyzer using YOLOv8 for vehicle/pedestrian detection, ByteTrack-based tracking, speed estimation, and traffic density classification.

A real-time computer vision system that analyzes traffic video footage 
using YOLOv8 for object detection and ByteTrack for multi-object tracking.

## Features
- Detects and classifies vehicles (cars, motorcycles, buses, trucks) and pedestrians
- Tracks objects across frames with persistent IDs
- Estimates real-world speed (km/h) for each tracked object
- Classifies traffic density per frame (Low / Medium / High)
- Applies stricter confidence and shape-based filtering for pedestrian detection 
  to reduce false positives
- Generates post-run analytics: vehicle and pedestrian count graphs over time

## Tech Stack
Python, OpenCV, Ultralytics YOLOv8, PyTorch, Matplotlib

## Status
Core detection, tracking, and analytics pipeline complete. 
Planned additions: traffic chatbot, adaptive signal control.

## How It Works
1. Loads a pretrained YOLOv8s model (COCO-trained)
2. Reads frames from a traffic video
3. Runs detection + tracking with class-specific confidence thresholds
4. Estimates speed using pixel displacement over a frame history window
5. Overlays bounding boxes, labels, and density status on the video
6. Plots vehicle/pedestrian counts per frame after the run
