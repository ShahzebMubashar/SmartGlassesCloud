from ultralytics import YOLO
import gradio as gr
import os
import numpy as np
import cv2

# Load model
model = YOLO('best_v3.pt')
THRESHOLD = 0.2  # Set back to 0.5 for stability, or keep 0.1 for testing
last_spoken = ""

def main(frame):
    global last_spoken
    
    if frame is None:
        return ""

    try:
        # ── Step 1: Image Processing ────────────────────
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
                
                if conf >= THRESHOLD:
                    detected.add(label)

        # ── Step 3: Filtering logic ──────────────────────
        if not detected:
            return "" 

        text = ", ".join(sorted(detected))

        # Only return text if the list of objects has changed
        if text == last_spoken:
            return "" 

        last_spoken = text
        print(f"✅ Success: Sending text to local: {text}")
        return text 

    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return ""

# Interface Setup (Single Output: Textbox)
demo = gr.Interface(
    fn=main,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(label="Detected Text"),
    title="BlindAid - Text-only API"
)

if __name__ == "__main__":
    demo.launch()