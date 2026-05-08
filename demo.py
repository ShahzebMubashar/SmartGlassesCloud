from ultralytics import YOLO
import gradio as gr
from gtts import gTTS
import tempfile
import os
import numpy as np
import cv2
import time
from collections import Counter
import pytesseract
import easyocr

# --- CONFIGURATION ---
CUSTOM_MODEL_PATH = "best_v3.pt"  # Updated to match your HF file structure
model = YOLO(CUSTOM_MODEL_PATH)
THRESHOLD = 0.5
last_spoken = set()

# Initialize OCR
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader

# --- UTILITY FUNCTIONS ---
def get_spatial_phrase(det, frame_width):
    x1, _, x2, _ = det["bbox"]
    center_x = (x1 + x2) / 2.0
    if center_x < frame_width / 3:
        return "on the left"
    if center_x > (2 * frame_width) / 3:
        return "on the right"
    return "ahead"

def estimate_distance_m(det, frame_shape):
    x1, y1, x2, y2 = det["bbox"]
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = max(1e-6, box_area / frame_area)
    dist = min(6.0, max(0.4, 1.35 / (ratio ** 0.5)))
    return dist

def run_ocr_hybrid(frame):
    try:
        # Tesseract Path
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tesseract_text = pytesseract.image_to_string(thr, lang="eng", config="--psm 6").strip()
        
        if len(tesseract_text) > 10:
            return tesseract_text
            
        # EasyOCR Fallback
        reader = get_easyocr_reader()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chunks = reader.readtext(rgb, detail=0, paragraph=True)
        return " ".join(chunks)[:260] if chunks else "No text found."
    except:
        return "OCR Error."

# --- MAIN GRADIO FUNCTION ---
def main(frame, mode):
    global last_spoken

    if frame is None:
        return None, "Please provide an image."

    # Handle filepath vs numpy
    if isinstance(frame, str):
        frame = cv2.imread(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. OCR MODE
    if mode == "OCR (Read Text)":
        text_found = run_ocr_hybrid(frame)
        tts = gTTS(text=text_found, lang='en')
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name
        tts.save(output_path)
        return output_path, text_found

    # 2. OBJECT DETECTION MODE
    results = model(frame, verbose=False)[0]
    detections = []
    detected_labels = set()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])
        if conf >= THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"label": label, "bbox": [x1, y1, x2, y2], "conf": conf})
            detected_labels.add(label)

    if not detections:
        return None, "No objects detected."

    # Build Alert String (Priority Logic)
    nearest = min(detections, key=lambda d: estimate_distance_m(d, frame.shape))
    dist = estimate_distance_m(nearest, frame.shape)
    where = get_spatial_phrase(nearest, frame.shape[1])
    alert_text = f"{nearest['label']} {where}, around {dist:.1f} meters."

    # Prevent repeating the exact same detection group
    if detected_labels == last_spoken:
        return None, "Scene unchanged: " + alert_text
    
    last_spoken = detected_labels

    # Generate Audio
    tts = gTTS(text=alert_text, lang='en')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name
    tts.save(output_path)

    return output_path, alert_text

# --- GRADIO INTERFACE ---
demo = gr.Interface(
    fn=main,
    inputs=[
        gr.Image(type="filepath", label="Webcam Stream"),
        gr.Radio(["Object Detection", "OCR (Read Text)"], value="Object Detection", label="Mode")
    ],
    outputs=[
        gr.Audio(autoplay=True, label="Voice Alert"),
        gr.Textbox(label="Status/Text Found")
    ],
    title="BlindAid - Object & Text Guidance",
    description="Custom YOLOv8 assistive vision. OCR mode reads text, Object Detection provides distance/direction.",
    live=True
)