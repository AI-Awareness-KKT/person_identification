import os
import time
import base64
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# ===============================
# HARD MEMORY CONTROLS (CRITICAL)
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

torch.set_grad_enabled(False)

app = Flask(__name__)

# ===============================
# LOAD MODELS ONCE (GLOBAL)
# ===============================
yolo_model = YOLO("yolo11n.pt")
yolo_model.fuse = False  # prevents internal layer fusion OOM

age_processor = AutoImageProcessor.from_pretrained(
    "nateraw/vit-age-classifier",
    use_fast=True
)
age_model = AutoModelForImageClassification.from_pretrained(
    "nateraw/vit-age-classifier"
)

gender_processor = AutoImageProcessor.from_pretrained(
    "rizvandwiki/gender-classification",
    use_fast=True
)
gender_model = AutoModelForImageClassification.from_pretrained(
    "rizvandwiki/gender-classification"
)

age_ranges = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"
]

# ===============================
# INFERENCE THROTTLING (KEY)
# ===============================
LAST_INFER_TIME = 0
INFER_INTERVAL = 2.5  # seconds (safe for free tier)
CACHED_RESULT = []

# ===============================
# HELPERS
# ===============================
def estimate_age(face):
    img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    inputs = age_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = age_model(**inputs).logits
        idx = logits.argmax().item()
        start, end = map(int, age_ranges[idx].split("-"))
        return str((start + end) // 2)

def detect_gender(face):
    img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    inputs = gender_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = gender_model(**inputs).logits
        return "Male" if logits.argmax().item() == 1 else "Female"

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global LAST_INFER_TIME, CACHED_RESULT

    try:
        now = time.time()

        # Decode image
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Resize early (IMPORTANT)
        frame = cv2.resize(frame, (640, 480))

        # ===============================
        # THROTTLE HEAVY INFERENCE
        # ===============================
        if now - LAST_INFER_TIME < INFER_INTERVAL:
            return jsonify(CACHED_RESULT)

        LAST_INFER_TIME = now
        detections = []

        results = yolo_model.predict(
            source=frame,
            conf=0.6,
            classes=[0],
            imgsz=640,
            verbose=False
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            if len(boxes) > 0:
                # ONLY ONE PERSON (largest box)
                x1, y1, x2, y2 = max(
                    boxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )

                person = frame[y1:y2, x1:x2]

                if person.size > 0:
                    age = estimate_age(person)
                    gender = detect_gender(person)

                    detections.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "age": age,
                        "gender": gender
                    })

        CACHED_RESULT = detections

        # Explicit cleanup
        del frame, results
        torch.cuda.empty_cache()

        return jsonify(CACHED_RESULT)

    except Exception as e:
        print("ANALYZE ERROR:", e)
        return jsonify(CACHED_RESULT)

@app.route("/favicon.ico")
def favicon():
    return "", 204

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
