from flask import Flask, render_template, Response
import cv2
import threading
import logging

# Initialize Flask app
app = Flask(__name__)

# Logging Configuration
logging.basicConfig(
    filename="detection_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global variables for video streaming
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera
is_streaming = False

# Function to generate video frames
def generate_frames():
    global is_streaming
    is_streaming = True
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the dashboard
@app.route('/')
def home():
    logs = read_logs()
    return render_template("dashboard.html", logs=logs)

# Function to read logs
def read_logs():
    with open("detection_logs.txt", "r") as file:
        logs = file.readlines()
    return logs

# Function to stop streaming
def stop_streaming():
    global is_streaming
    is_streaming = False
    camera.release()

# Run the Flask app
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        stop_streaming()