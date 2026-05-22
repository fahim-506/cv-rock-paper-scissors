# ✊✋✌️ Rock Paper Scissors — Real-Time Detection with YOLOv8

A real-time hand gesture detection system that recognizes **Rock**, **Paper**, and **Scissors** using a custom-trained **YOLOv8n** model and a live webcam feed.

![Demo](https://s5.ezgif.com/tmp/ezgif-56371dd32f5ef249.gif)

---

## 📁 Project Structure

```
ROCK-PAPER-SCISSORS/
├── cv-rps/                        # Python virtual environment
├── notebook/
│   └── Rock-Paper-Scissor.ipynb   # Training notebook (Google Colab)
├── .gitignore
├── best.pt                        # Best trained model weights — used for inference
├── LICENSE
├── README.md
├── requirements.txt               # Python dependencies
└── rock-paper-scissor.py          # Real-time inference script
```

---

## 🚀 How It Works

```
Webcam Feed (OpenCV)
       │
       ▼
  Frame Capture
       │
       ▼
 YOLOv8n Inference  ◄──── best.pt (trained weights)
       │
       ▼
Confidence Filter (>= 0.75)
       │
       ▼
Bounding Box + Label Annotation
       │
       ▼
 Live Display Window
```

1. **Webcam** captures live video frames using **OpenCV** (`cv2.VideoCapture`).
2. Each frame is fed into the **YOLOv8n** model loaded from `best.pt` for inference.
3. Detections with confidence **>= 0.75** are kept; the rest are filtered out.
4. YOLOv8 annotates the frame with **bounding boxes** and **class labels** (Rock / Paper / Scissors).
5. The annotated frame is rendered in a **real-time display window**.

---

## 🧠 Model Training

| Detail        | Info                                                                 |
|---------------|----------------------------------------------------------------------|
| **Model**     | YOLOv8n (nano)                                                       |
| **Dataset**   | [Rock Paper Scissors – Roboflow Universe](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw) |
| **Platform**  | Google Colab                                                         |
| **GPU**       | NVIDIA T4                                                            |
| **Notebook**  | `notebook/Rock-Paper-Scissor.ipynb`                                  |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/fahim-506/cv-rock-paper-scissors.git
cd cv-rock-paper-scissors
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
ipykernel
ultralytics
opencv-python
```

---

## ▶️ Run the Project

```bash
python rock-paper-scissor.py
```

- Make sure your **webcam is connected**.
- The model file `best.pt` must be in the project root directory.
- Press **`q`** to quit the detection window.

---

## 🖥️ Inference Script Overview

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
THRESHOLD = 0.75

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=THRESHOLD)
    annotated_frame = results[0].plot()

    cv2.imshow("Rock Paper Scissors – YOLOv8", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📦 Dataset

The model was trained on the **Rock Paper Scissors** dataset from Roboflow:

🔗 [https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.