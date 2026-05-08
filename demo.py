from ultralytics import YOLO
import gradio as gr
from gtts import gTTS
import tempfile
import os

model = YOLO('best_v3.pt')
THRESHOLD = 0.5
last_spoken = set()

def main(frame):
    global last_spoken
    
    if frame is None:
        return None

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

    tts = gTTS(text=text, lang='en')               # ← gTTS generates speech
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name
    tts.save(output_path)                          # ← saves to temp file

    return output_path                             # ← sends audio back to browser

demo = gr.Interface(
    fn=main,
    inputs=gr.Image(
        sources="webcam",
        streaming=True,
        type="numpy"
    ),
    outputs=gr.Audio(autoplay=True),
    title="BlindAid - Object Detection",
    live=True
)
  