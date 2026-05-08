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
        return "" # Return empty string for "no input"

    try:
        # Step 1: Image Processing (Keep this as is)
        if isinstance(frame, str):                  
            frame = cv2.imread(frame)               
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 2: YOLO Inference (Keep this as is)
        results = model(frame)
        detected = set()
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf >= THRESHOLD:
                    detected.add(label)

        # Step 3: Text Generation & Filtering
        if not detected:
            return "" # Return empty string for "no objects"

        # Sort alphabetically so the string is consistent for comparison
        text = ", ".join(sorted(detected))

        # Only return the string if it's DIFFERENT from the last one
        if text == last_spoken:
            return "" 

        last_spoken = text
        print(f"📡 API Sending: {text}")
        return text 

    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return ""

# Change your interface to this:
demo = gr.Interface(
    fn=main,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(), # Simplified to just one output
    api_name="main"       # Explicitly name the endpoint for the Pi
)