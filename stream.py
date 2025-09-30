from flask import Flask, Response
import cv2

# Initialize Flask app and camera
app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 for default webcam, adjust if necessary

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Route to provide the video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Visit /video_feed to view the live stream."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

