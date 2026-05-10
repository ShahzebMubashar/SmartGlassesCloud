import cv2
import os
import time
import tempfile
import httpx
import pyttsx3
from gradio_client import Client, handle_file

# --- CONFIGURATION ---
HF_SPACE_URL = "aliyanFYP/fyp"
# Pi Camera via libcamera usually exposes V4L2 as index 0 or 10 — script tries 0..2.
CAMERA_INDICES = (0, 1, 2)
# 3 seconds is ideal for Pi 3 to maintain stability
DELAY_BETWEEN_CALLS = 3
# Warn if OpenCV never gets a frame (rpicam can work while cv2.read() does not).
_FRAME_FAIL_WARN_SEC = 15

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


def _open_cv_capture():
    """OpenCV + Pi Camera: prefer V4L2; probe until we get a real frame."""
    v4l2 = getattr(cv2, "CAP_V4L2", None)

    def try_open(idx, api):
        cap_local = (
            cv2.VideoCapture(idx, api)
            if api is not None
            else cv2.VideoCapture(idx)
        )
        if not cap_local.isOpened():
            cap_local.release()
            return None
        cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frm = cap_local.read()
        if ret and frm is not None and frm.size > 0:
            return cap_local
        cap_local.release()
        return None

    for idx in CAMERA_INDICES:
        if v4l2 is not None:
            cap = try_open(idx, v4l2)
            if cap is not None:
                print(f"✅ Camera OK via V4L2, index {idx}.", flush=True)
                return cap
        cap = try_open(idx, None)
        if cap is not None:
            print(f"✅ Camera OK (default backend), index {idx}.", flush=True)
            return cap

    print(
        "❌ OpenCV could not capture frames from indices "
        f"{CAMERA_INDICES}. rpicam-hello may still work — install "
        "`sudo apt install v4l-utils` and run `v4l2-ctl --list-devices`, "
        "then set CAMERA_INDICES to match /dev/videoN.",
        flush=True,
    )
    return None


def run_detector():
    cap = _open_cv_capture()
    if cap is None:
        return

    # Starting mode (Match the Radio options in demo.py)
    current_mode = "Object Detection"

    print(f"🚀 Pi BlindAid Active in {current_mode} mode.")
    print("Hold an object to hear distance, or trigger OCR via API if needed.")
    print("(No camera window — headless mode. First API call can take several minutes.)")
    print(
        "Network: if `curl` to the Space URL shows 000, fix Wi‑Fi/Ethernet/DNS — "
        "API calls will fail or hang.\n",
        flush=True,
    )

    try:
        fail_started = None
        while True:
            ret, frame = cap.read()
            if not ret:
                now = time.monotonic()
                if fail_started is None:
                    fail_started = now
                elif now - fail_started >= _FRAME_FAIL_WARN_SEC:
                    print(
                        "⚠️ OpenCV is not receiving frames (another app may be "
                        "using the camera). Close rpicam/other capture and retry.",
                        flush=True,
                    )
                    fail_started = now
                time.sleep(1)
                continue

            fail_started = None

            # Save frame to temp file with compression to save bandwidth
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            try:
                # Call Hugging Face API with the current mode
                # Result[1] contains the string we want to speak
                print("📤 Sending frame to Hugging Face…", flush=True)
                t0 = time.monotonic()
                result = client.predict(
                    frame=handle_file(temp_path),
                    mode=current_mode,
                    api_name="/main"
                )
                dt = time.monotonic() - t0
                print(f"📥 Reply in {dt:.1f}s", flush=True)

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