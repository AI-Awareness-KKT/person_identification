from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import base64

app = Flask(__name__)

# ======================
# Load models ONCE (SAFE)
# ======================
model = YOLO("yolo11n.pt")
model.fuse = False   # 🔥 CRITICAL FIX

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

# ======================
# Helper functions
# ======================
def estimate_age(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = age_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = age_model(**inputs).logits
        idx = logits.argmax().item()
        start, end = map(int, age_ranges[idx].split("-"))
        return str((start + end) // 2)

def detect_gender(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = gender_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = gender_model(**inputs).logits
        return "Male" if logits.argmax().item() == 1 else "Female"

# ======================
# Routes
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json["image"]
        image_bytes = base64.b64decode(data.split(",")[1])

        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        detections = []

        # 🔥 SAFE prediction call
        results = model.predict(
            source=frame,
            conf=0.6,
            classes=[0],
            imgsz=640,
            verbose=False
        )

        for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            person = frame[y1:y2, x1:x2]

            if person.size == 0:
                continue

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

        return jsonify(detections)

    except Exception as e:
        print("ANALYZE ERROR:", e)
        return jsonify([])

@app.route("/favicon.ico")
def favicon():
    return "", 204

# ======================
# Run
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
