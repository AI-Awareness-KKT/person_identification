import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, \
    AutoModelForImageClassification
import torch
from PIL import Image

# Initialize YOLOv8 model
model = YOLO('./yolo11n.pt')

# Initialize BLIP image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize DEX age detection model
age_processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier", use_fast=True)
age_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
age_ranges = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']

# Initialize gender detection model
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize dictionaries to store track information
track_counts = defaultdict(int)
track_captions = {}
track_ages = {}
track_genders = {}
track_frame_counts = defaultdict(int)
REFRESH_INTERVAL = 30

# Constants for detection
MIN_PERSON_HEIGHT = 200
CONFIDENCE_THRESHOLD = 0.65

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


def generate_caption(image):
    """Generate caption for a given image using BLIP model"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    inputs = processor(pil_image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def estimate_age(image):
    """Estimate age using DEX model"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Prepare image for the model
        inputs = age_processor(pil_image, return_tensors="pt")

        # Get prediction
        with torch.no_grad():
            outputs = age_model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            predicted_idx = predictions.argmax().item()
            confidence = predictions[0][predicted_idx].item()

            # Return the age range
            age_range = age_ranges[predicted_idx]
            # Return middle of the range for display
            start, end = map(int, age_range.split('-'))
            return f"~{(start + end) // 2}"

    except Exception as e:
        print(f"Age estimation error: {e}")
        return None


def detect_gender(image):
    """Detect gender using the gender classification model"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        inputs = gender_processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = gender_model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            predicted_idx = predictions.argmax().item()

            return "Male" if predicted_idx == 1 else "Female"

    except Exception as e:
        print(f"Gender detection error: {e}")
        return None


def draw_wrapped_text(frame, text, x, y, max_width, font, font_scale, color, thickness):
    """Draw text that wraps if it exceeds max_width"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        (line_width, _), _ = cv2.getTextSize(' '.join(current_line), font, font_scale, thickness)
        if line_width > max_width:
            if len(current_line) > 1:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(' '.join(current_line))
                current_line = []

    if current_line:
        lines.append(' '.join(current_line))

    line_height = cv2.getTextSize('A', font, font_scale, thickness)[0][1] + 5
    background_padding = 5

    # Calculate total height needed
    total_height = len(lines) * line_height

    # Draw background for all lines
    cv2.rectangle(frame,
                  (x - background_padding, y - total_height - background_padding),
                  (x + max_width + background_padding, y + background_padding),
                  (0, 255, 0),
                  -1)

    # Draw each line from bottom to top
    for i, line in enumerate(reversed(lines)):
        y_pos = y - (i * line_height)
        cv2.putText(frame,
                    line,
                    (x, y_pos),
                    font,
                    font_scale,
                    color,
                    thickness)

    return total_height


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
MAIN_FONT_SCALE = 2.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=1024
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = box
            height = y2 - y1
            width = x2 - x1

            if height < MIN_PERSON_HEIGHT:
                continue

            track_counts[track_id] += 1
            track_frame_counts[track_id] += 1

            # Generate caption, estimate age, and detect gender for new tracks or refresh existing ones
            if (track_id not in track_captions or
                    track_frame_counts[track_id] % REFRESH_INTERVAL == 0):

                person_image = frame[y1:y2, x1:x2]
                if person_image.size > 0:
                    caption = generate_caption(person_image)
                    track_captions[track_id] = caption

                    age = estimate_age(person_image)
                    track_ages[track_id] = age

                    gender = detect_gender(person_image)
                    track_genders[track_id] = gender

                    if track_id not in track_captions:
                        print(
                            f"New person detected (ID: {track_id}) - Age: {age} - Gender: {gender} - Caption: {caption}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            caption = track_captions.get(track_id, "Processing...")

            # Draw caption first
            caption_height = draw_wrapped_text(frame,
                                               caption,
                                               x1,
                                               y1 - 5,
                                               width,
                                               FONT,
                                               FONT_SCALE,
                                               (0, 0, 0),
                                               FONT_THICKNESS)

            # Draw ID, Age, and Gender below the caption
            top_label = f"ID: {track_id} | Age: {track_ages.get(track_id, 'Unknown')} | Gender: {track_genders.get(track_id, 'Unknown')}"
            (label_width, label_height), _ = cv2.getTextSize(top_label, FONT, FONT_SCALE, FONT_THICKNESS)

            # Position the label below the caption
            label_y = y1 - caption_height - 10
            cv2.rectangle(frame,
                          (x1, label_y - label_height - 5),
                          (x1 + label_width, label_y + 5),
                          (0, 255, 0),
                          -1)
            cv2.putText(frame,
                        top_label,
                        (x1, label_y),
                        FONT,
                        FONT_SCALE,
                        (0, 0, 0),
                        FONT_THICKNESS)

    # Draw total count
    total_count = len(track_counts)
    '''cv2.putText(frame,
                f"Total People: {total_count}",
                (10, 60),
                FONT,
                MAIN_FONT_SCALE,
                (0, 0, 255),
                FONT_THICKNESS + 1)'''

    cv2.imshow("YOLOv8 Tracking with Age, Gender and Captions", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Results:")
print(f"Total unique people detected: {len(track_counts)}")
print("\nPerson Descriptions:")
for track_id in track_counts.keys():
    print(
        f"Person {track_id}: Age    : {track_ages.get(track_id, 'Unknown')} | Gender: {track_genders.get(track_id, 'Unknown')} | {track_captions.get(track_id, 'No caption')}")