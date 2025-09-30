import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import pygame
import os

# Email Configuration
EMAIL_ADDRESS = "projecteyesafety@gmail.com"
EMAIL_PASSWORD = "dkxf bqxf phsk orob"
RECIPIENT_EMAIL = "rohit.kumar813044@gmail.com"

# Set up directories
save_dir = r"C:\Users\rohit\Pictures\weapon detection"
os.makedirs(save_dir, exist_ok=True)

# Load YOLO model
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize pygame mixer
pygame.mixer.init()

# Path to the MP3 file
sound_path = os.path.join(os.getcwd(), "beep_sound.mp3")
if not os.path.exists(sound_path):
    print(f"Error: MP3 file not found at {sound_path}")

# Cooldown time
cooldown_time = timedelta(seconds=10)
last_detection_time = datetime.min

def play_sound():
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        print("Playing sound...")
    except Exception as e:
        print(f"Error playing sound: {e}")

def start_detection():
    global last_detection_time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        height, width, channels = img.shape
        print(f"Frame dimensions: {height}x{width}")

        # Detecting weapons
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = "Weapon"
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

                # Check cooldown
                current_time = datetime.now()
                if current_time - last_detection_time > cooldown_time:
                    last_detection_time = current_time

                    timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')
                    img_path = os.path.join(save_dir, f"weapon_{timestamp}.jpg")
                    cv2.imwrite(img_path, img)
                    send_email(img_path)

                    # Play sound
                    play_sound()

        # Show the frame
        cv2.imshow("Weapon Detection", img)

        # Exit if 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def send_email(image_path):
    try:
        subject = "Weapon Detected - Project Eye"
        body = "A weapon has been detected. See the attached image."
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("Email sent successfully.")
    except Exception as e:
        print(f"Email failed: {e}")

if __name__ == "__main__":
    start_detection()