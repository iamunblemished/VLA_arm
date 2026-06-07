import cv2
import numpy as np
import subprocess
import os
import time
from flask import Flask, Response, send_file

app = Flask(__name__)

# --- HARDWARE & VLM CONFIG ---
WIDTH, HEIGHT = 1920, 1080
FRAME_SIZE = WIDTH * HEIGHT
PORT = 5000
TARGET_FPS = 15

# --- FINAL COLOR CALIBRATION ---
R_GAIN = 1.15
G_GAIN = 0.95  # Slightly mutes the green
B_GAIN = 1.25
GAMMA = 1.4    # Brightens shadows (1.0 is neutral)

# Pre-calculate Gamma Table to save CPU
GAMMA_LUT = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]).astype("uint8")

# Cleanup previous sessions
os.system(f"sudo fuser -k {PORT}/tcp > /dev/null 2>&1")
os.system("sudo fuser -k /dev/video0 > /dev/null 2>&1")

# Global for the latest frame (used for the /capture endpoint)
latest_frame = None

def get_raw_frames():
    global latest_frame
    cmd = [
        "v4l2-ctl", "-d", "/dev/video0",
        "--set-fmt-video=width=1920,height=1080,pixelformat=RGGB",
        "--stream-mmap", "--stream-to=-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    last_time = 0

    while True:
        if (time.time() - last_time) < (1.0 / TARGET_FPS):
            time.sleep(0.01)
            continue
            
        raw_data = process.stdout.read(FRAME_SIZE)
        if len(raw_data) < FRAME_SIZE: continue
        last_time = time.time()

        # 1. Debayer
        raw_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH))
        bgr_frame = cv2.cvtColor(raw_array, cv2.COLOR_BayerBG2BGR)
        
        # 2. Downscale (Save CPU)
        small = cv2.resize(bgr_frame, (640, 480))
        
        # 3. Apply Color Gains & Gamma
        temp = small.astype(np.float32)
        temp[:, :, 0] *= B_GAIN
        temp[:, :, 1] *= G_GAIN
        temp[:, :, 2] *= R_GAIN
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        
        # Apply pre-calculated Gamma LUT
        latest_frame = cv2.LUT(temp, GAMMA_LUT)
        
        ret, buffer = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(get_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    # Endpoint to save a high-quality frame for VLM inference
    if latest_frame is not None:
        cv2.imwrite("vlm_input.jpg", latest_frame)
        return send_file("vlm_input.jpg", mimetype='image/jpeg')
    return "No frame captured", 500

@app.route('/')
def index():
    return "<h1>Radxa VLA Vision</h1><img src='/video_feed' width='640'><p><a href='/capture'>Capture Frame for VLM</a></p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, threaded=True)
