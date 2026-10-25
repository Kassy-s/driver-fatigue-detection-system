# Real-Time Driver Fatigue & Distraction Detection System

An edge computer vision pipeline engineered to detect real-time driver drowsiness, micro-sleep events, and yawning patterns using facial landmark geometric analysis.

---

## Technical Architecture

[Live Video Stream / Camera Frame]
│
▼
[MediaPipe FaceMesh Pipeline]
(468 3D Ocular & Facial Nodes)
│
┌─────────┴─────────┐
▼                   ▼
[Ocular Points]     [Mouth Points]
(Indices 33-387)    (Indices 13-308)
│                   │
▼                   ▼
[Eye Aspect Ratio]  [Mouth Aspect Ratio]
│                   │
└─────────┬─────────┘
▼
[State Decision Engine]
• Consecutive Frame Counter
• Dynamic Alert Triggering
│
▼
[Overlay Display & Audit Log]

---

## Mathematical Formulation

### 1. Eye Aspect Ratio (EAR)
Quantifies ocular openness using horizontal and vertical landmark distances:

$$\text{EAR} = \frac{\Vert{}p_2 - p_6\Vert{} + \Vert{}p_3 - p_5\Vert{}}{2 \cdot \Vert{}p_1 - p_4\Vert{}}$$

* Prolonged drop below $\text{EAR} \le 0.22$ across consecutive frames triggers a drowsiness alert.

### 2. Mouth Aspect Ratio (MAR)
Evaluates vertical mouth distension to detect yawning behavior:

$$\text{MAR} = \frac{\Vert{}p_{\text{top}} - p_{\text{bottom}}\Vert{}}{\Vert{}p_{\text{left}} - p_{\text{right}}\Vert{}}$$

---

## Project Structure

text
├── reports/          # Session audit logs and JSON diagnostics
├── src/
│   ├── detector.py   # FaceMesh integration and state manager
│   └── metrics.py    # Geometric Euclidean formulations (EAR/MAR)
├── main.py           # Application entry point with dual live/benchmark mode
├── requirements.txt  # Project dependencies (OpenCV, MediaPipe, NumPy)
└── README.md

---

## Getting Started

### Prerequisites
* Python 3.9+
* Standard webcam / integrated camera

### Installation & Run
1. Clone the repository:
bash
git clone https://github.com//driver-fatigue-detection-system.git
cd driver-fatigue-detection-system

2. Install dependencies:
bash
pip install -r requirements.txt

3. Start monitoring:
bash
python main.py

*(Press `q` inside the video stream window to end the session).*

---

## License
Distributed under the MIT License.