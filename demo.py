from ultralytics import YOLO
import gradio as gr
from gtts import gTTS
import tempfile
import os
import numpy as np
import cv2

# Load model
model = YOLO('best_v3.pt')
THRESHOLD = 0.1
last_spoken = set()

def main(frame):
    global last_spoken
    
    if frame is None:
        return None, None

    try:
        # ── Step 1: Image Processing ────────────────────
        # Gradio sends a filepath string when type="filepath"
        if isinstance(frame, str):                  
            frame = cv2.imread(frame)               
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Step 2: Model Inference ─────────────────────
        results = model(frame)
        detected = set()

        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # Debug print for HF logs
                print(f"Detected: {label} ({conf:.2f})")

                if conf >= THRESHOLD:
                    detected.add(label)

        # ── Step 3: Filtering logic ──────────────────────
        if not detected:
            print("ℹ️ Nothing detected above threshold.")
            return None, None 

        if detected == last_spoken:
            # Important: still return None to keep test.py silent
            return None, None 

        last_spoken = detected
        text = ", ".join(detected)

        # ── Step 4: TTS Generation ───────────────────────
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name
        tts.save(output_path)
        
        print(f"✅ Success: Sending audio and text: {text}")
        return output_path, text 

    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return None, None

# Interface Setup
demo = gr.Interface(
    fn=main,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Audio(label="Voice Alert"),    # Output 1 (result[0])
        gr.Textbox(label="Detected Text") # Output 2 (result[1])
    ],
    title="BlindAid - Object Detection"
)

# This is required for app.py to work
if __name__ == "__main__":
    demo.launch()