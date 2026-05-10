import cv2
import os
import time
import tempfile
import pyttsx3 
from gradio_client import Client, handle_file

# --- CONFIGURATION ---
HF_SPACE_URL = "aliyanFYP/fyp"
client = Client(HF_SPACE_URL)

# Initialize Local TTS
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    if text and text.strip():
        print(f"📢 AI: {text}")
        engine.say(text)
        engine.runAndWait()

def run_detector():
    cap = cv2.VideoCapture(0)
    # Recommended resolution for stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # NEW: Toggle this to "OCR (Read Text)" to test reading mode
    current_mode = "Object Detection" 

    print(f"✅ Started in {current_mode} mode.")
    print("Press 'o' to toggle OCR, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Save frame to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)

            try:
                # UPDATED: We now send TWO arguments (image and mode)
                # Result[1] is the text string from the Textbox
                result = client.predict(
                    frame=handle_file(temp_path),
                    mode=current_mode,
                    api_name="/main"
                )

                # Accessing result[1] because result[0] is the audio file path
                status_text = result[1]

                if status_text and "No objects" not in status_text:
                    speak(status_text)
                else:
                    print("Scanning...")

            except Exception as e:
                print(f"📡 API Error: {e}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # GUI and Controls
            cv2.putText(frame, f"Mode: {current_mode} (Press 'o' to toggle)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('BlindAid Testing', frame)
            
            key = cv2.waitKey(2000) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                current_mode = "OCR (Read Text)" if current_mode == "Object Detection" else "Object Detection"
                print(f"🔄 Switched to {current_mode}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detector()