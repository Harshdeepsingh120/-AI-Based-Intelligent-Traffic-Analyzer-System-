# AI‑Based Intelligent Traffic Analyzer

**A lightweight, end‑to‑end traffic‑analysis toolkit** that ingests a raw traffic video, runs YOLOv8 object detection, SORT multi‑object tracking, lane‑wise density counting, speed estimation, and an adaptive signal‑control state machine. Results are visualised in a clean web dashboard and can be queried via an integrated hybrid chatbot.

---

## Architecture Overview

| Component | Description |
|---|---|
| **Detection** (`ultralytics` YOLOv8) | Detects `car`, `motorcycle`, `bus`, `truck`, `person` objects per frame. |
| **Tracking** (`tracker.py` – custom SORT) | Associates detections across frames using a 7‑D Kalman filter and Hungarian matching. |
| **Speed Estimation** (`speed_estimator.py`) | Calibrates meters‑per‑pixel from the first 100 frames and estimates per‑vehicle speed using a K‑frame displacement formula. |
| **Signal Control** (`signal_control.py`) | Computes left/right lane densities, decides the active signal phase, and outputs a dynamic green‑time duration. |
| **Chatbot** (`chatbot.py`) | Hybrid tier: primary Gemini AI query handling, fallback rule‑based intent classification for count, signal, speed, density, etc. |
| **Web Dashboard** (FastAPI + static HTML/CSS/JS) | Thin API layer (`app.py`) wraps the core pipeline and serves a minimal UI with upload, live feed, metrics, and chat drawer. |

---

## How the System Works (Data Flow)
1. **Video Upload** – The user drags‑and‑drops an `.mp4/.avi/.mov` file into the dashboard.
2. **Frame Pre‑processing** – The backend reads frames with OpenCV, skips every second frame to keep latency low.
3. **YOLOv8 Detection** – Each processed frame is sent to the YOLO model (`yolov8n.pt`). Class‑specific confidence thresholds are applied (car 0.45, bus/truck/motorcycle 0.25, person 0.55).
4. **SORT Tracking** – Detections are fed to the custom SORT implementation, which returns a list of tracked objects `(bbox, track_id, class_name)`.
5. **Lane Assignment** – The frame midpoint separates left/right lanes; vehicles are counted per lane.
6. **Speed Estimation** – For each tracked vehicle, the `SpeedEstimator` updates a K‑frame displacement and converts pixel movement to km/h using the calibrated `meters_per_pixel` value.
7. **Signal‑Control State Machine** – Every `t_eval` seconds (`5 s`) the lane densities are compared; the controller updates the active signal (`GREEN_LEFT`, `GREEN_RIGHT`, or `ALL_RED`) and computes the next green‑time based on the ratio of densities.
8. **Live Dashboard Update** – The backend writes an annotated frame (`processed_frame.jpg`) and updates a shared JSON state. The frontend polls `/api/status` to refresh the progress bar, metrics cards, signal panel, and video feed.
9. **Chatbot Query Handling** – Users can open the chat drawer and ask natural‑language questions (e.g., "How many cars were detected?"). The request is routed to `chatbot.py`, which first tries the Gemini API (requires `GEMINI_API_KEY`). If the key is missing or the call fails, a fast rule‑based fallback provides deterministic answers based on the current state.

---

## Setup Instructions
1. **Python** – 3.11 or newer (tested with 3.12). 
2. **Clone / download the repository** into a directory of your choice.
3. **Create a virtual environment** (recommended) and activate it:
   ```
   # Windows PowerShell
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install dependencies** (exact versions are pinned in `requirements.txt`):
   ```
   pip install -r requirements.txt
   ```
5. **Download the YOLOv8 model weights** (`yolov8n.pt`).
   - Visit https://github.com/ultralytics/ultralytics/releases/tag/v8.4.41 or run:
     ```
     wget https://github.com/ultralytics/assets/releases/download/v8.4.41/yolov8n.pt -O yolov8n.pt
     ```
   - Place `yolov8n.pt` in the project root (same folder as `app.py`).
6. **Gemini API key** (optional, for AI‑tier chatbot):
   - Obtain a key from https://ai.google.dev/gemini-api.
   - Set an environment variable:
     ```
     # Windows PowerShell
     $Env:GEMINI_API_KEY="YOUR_KEY_HERE"
     # macOS / Linux
     export GEMINI_API_KEY="YOUR_KEY_HERE"
     ```
   - Alternatively, create a `.env` file in the project root containing `GEMINI_API_KEY=YOUR_KEY_HERE`. The `python-dotenv` package automatically loads it.

---

## Running the Application
### Option A – One‑click (Windows)
Double‑click `run.bat`. It will:
* Install missing Python packages (fast‑api, uvicorn, etc.)
* Launch the FastAPI server on **port 8000**
* Open your default browser at `http://localhost:8000`

### Option B – Manual (cross‑platform)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open a browser and navigate to `http://localhost:8000`.

---

## Running the Evaluation Suite
The script `evaluation.py` evaluates detection, tracking, and speed‑estimation against a small hand‑annotated ground‑truth set (20 frames).
```bash
python evaluation.py
```
**Expected file layout** (relative to the project root):
* `ground_truth.json` (or `base_gt.json`) – contains per‑frame GT boxes and class IDs.
* `traffic.mp4` – a sample video used for evaluation.
The script prints a summary table (see below).

---

## Measured Evaluation Results
| Metric | Measured Result |
|---|---|
| **Overall Detection Precision** | **99.1 %** |
| **Overall Detection Recall** | **55.7 %** |
| **Overall Detection mAP@0.50** | **55.2 %** |
| **Tracking MOTA** | **50.7 %** |
| **Tracking MOTP** | **92.4 %** |
| **Tracking ID Switches** | **9** |
| **Speed Estimation MAE** | **1.47 km/h** |
| **Speed Estimation MAPE** | **7.2 %** |
| **Signal Response Latency (mean)** | **4.50 s** |
| **Chatbot Combined Accuracy** | **100 %** |

**Per‑Class Detection Breakdown**
* **car** – Precision **98.9 %**, Recall **68.9 %** (135 GT)
* **bus** – Precision **100 %**, Recall **100 %** (10 GT)
* **truck** – Precision **100 %**, Recall **66.7 %** (12 GT)
* **person** – Precision **100 %**, Recall **4.4 %** (45 GT)
* **motorcycle** – Precision **0 %**, Recall **0 %** (1 GT)

---

## Known Limitations
* **Pedestrian recall** is deliberately low (≈ 4 %) because the current confidence gate (`0.55`) is required to avoid false positives on vertical street‑light poles.
* **Motorcycle detection** fails entirely on our test set – YOLOv8n lacks sufficient training samples for the surveillance angle used. A larger model or additional fine‑tuning would be needed.
* **Tracking MOTA** (≈ 51 %) is below the ideal 60 % range due to occasional ID switches during dense traffic; the custom SORT parameters have been tuned for speed rather than maximal accuracy.
* The web UI streams a single annotated frame image (`processed_frame.jpg`) rather than a true video stream to keep the backend lightweight. For higher‑throughput use‑cases you would replace it with an MJPEG or WebSocket stream.

---

## Tech Stack
| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, python‑multipart, requests |
| **Computer Vision** | Ultralytics YOLOv8 (`ultralytics`), OpenCV (`opencv‑python`), NumPy, SciPy, Matplotlib (optional visualisation) |
| **Tracking** | Custom SORT implementation (uses SciPy for Hungarian matching) |
| **Speed / Calibration** | NumPy, custom geometry utilities |
| **Signal Controller** | Pure Python state machine |
| **Chatbot** | Gemini API (`google‑generativeai` via `requests`), fallback rule‑based logic |
| **Web Frontend** | Plain HTML + CSS + Vanilla JS (no framework) |
| **Evaluation** | Custom `evaluation.py` script, uses `matplotlib` for optional plots |

---

## License & Contribution
This repository is provided under the MIT License. Feel free to fork, extend the detection model, or replace the rule‑based chatbot with a more sophisticated dialogue system.

---

*Happy coding!*
