from ultralytics import YOLO
import gradio as gr
from gtts import gTTS
import tempfile
import os
import numpy as np
import cv2

model = YOLO('best_v3.pt')
THRESHOLD = 0.5
last_spoken = set()

def main(frame):
    global last_spoken

    if frame is None:
        return None

    # ── handle both filepath and numpy array ────────────────────
    if isinstance(frame, str):                  # ← if filepath received
        frame = cv2.imread(frame)               # ← read it as numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame)
    detected = set()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])

            if confidence >= THRESHOLD:
                detected.add(label)

    if not detected or detected == last_spoken:
        return None

    last_spoken = detected

    text = ", ".join(detected)

    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name
    tts.save(output_path)

    return output_path

demo = gr.Interface(
    fn=main,
    inputs=gr.Image(
        sources="webcam",
        streaming=True,
        type="filepath"         # ← changed from "numpy" to "filepath"
    ),
    outputs=gr.Audio(autoplay=True),
    title="BlindAid - Object Detection",
    live=True
)