import cv2
import os
import time
import tempfile
import httpx
import pyttsx3
from gradio_client import Client, handle_file

# --- CONFIGURATION ---
HF_SPACE_URL = "aliyanFYP/fyp"
CAMERA_INDEX = 0 
# 3 seconds is ideal for Pi 3 to maintain stability
DELAY_BETWEEN_CALLS = 3 

# Initialize Local TTS (Optimized for Pi/Linux)
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 145) 
    engine.setProperty('volume', 1.0)
except Exception as e:
    print(f"TTS Init Error: {e}")
    engine = None

# Gradio Client hits HF over HTTPS; Pi 3 + Wi‑Fi often needs long timeouts.
_HF_TIMEOUT = httpx.Timeout(180.0, connect=90.0)


def _connect_hf_client():
    last_err = None
    for attempt in range(1, 4):
        try:
            return Client(HF_SPACE_URL, httpx_kwargs={"timeout": _HF_TIMEOUT})
        except Exception as e:
            last_err = e
            print(f"Hugging Face connect attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(8)
    raise RuntimeError(
        "Could not load Hugging Face Space config. Check Wi‑Fi, try ethernet or "
        "hotspot, or wait and retry."
    ) from last_err


client = _connect_hf_client()

def speak(text):
    if engine and text.strip():
        print(f"📢 AI: {text}")
        engine.say(text)
        engine.runAndWait()

def run_detector():
    # Use headless-friendly capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Pi 3 optimization: Use 640x480 resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Starting mode (Match the Radio options in demo.py)
    current_mode = "Object Detection" 

    if not cap.isOpened():
        print("❌ Error: Camera not found.")
        return

    print(f"🚀 Pi BlindAid Active in {current_mode} mode.")
    print("Hold an object to hear distance, or trigger OCR via API if needed.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            # Save frame to temp file with compression to save bandwidth
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            try:
                # Call Hugging Face API with the current mode
                # Result[1] contains the string we want to speak
                result = client.predict(
                    frame=handle_file(temp_path),
                    mode=current_mode,
                    api_name="/main"
                )

                if result and len(result) > 1:
                    status_text = result[1]
                    # Filter out empty or "No objects" messages to keep it quiet
                    if status_text and "No " not in status_text:
                        speak(status_text)
                else:
                    print("... scanning ...")

            except Exception as e:
                print(f"📡 API Connection Error: {e}")

            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            time.sleep(DELAY_BETWEEN_CALLS)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        cap.release()

if __name__ == "__main__":
    run_detector()