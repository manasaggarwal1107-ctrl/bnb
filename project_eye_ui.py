import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
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

# Global variables for UI
is_detection_running = False
cap = None

# Function to play sound
def play_sound():
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        print("Playing sound...")
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to send email
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
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# Function to start detection
def start_detection():
    global is_detection_running, cap, last_detection_time
    is_detection_running = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access the camera.")
        return

    while is_detection_running:
        ret, img = cap.read()
        if not ret:
            messagebox.showerror("Error", "Unable to read frame.")
            break

        height, width, channels = img.shape

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
                    if send_email(img_path):
                        email_status_label.config(text="Email Sent", fg="green")
                    else:
                        email_status_label.config(text="Email Failed", fg="red")

                    # Play sound
                    play_sound()

        # Convert image to RGB and display in UI
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        camera_label.config(image=img)
        camera_label.image = img

    cap.release()
    cv2.destroyAllWindows()

# Function to stop detection
def stop_detection():
    global is_detection_running
    is_detection_running = False
    if cap:
        cap.release()
    camera_label.config(image=None)

# Create the main window
root = tk.Tk()
root.title("Project Eye - Weapon Detection System")
root.geometry("1200x800")
root.configure(bg="#1e1e1e")

# Title Label
title_label = tk.Label(root, text="Project Eye", font=("Helvetica", 36, "bold"), fg="#00ff00", bg="#1e1e1e")
title_label.pack(pady=20)

# Camera Feed Label
camera_label = tk.Label(root, bg="#1e1e1e")
camera_label.pack()

# Detection Status Label
status_label = tk.Label(root, text="Status: Idle", font=("Helvetica", 18), fg="white", bg="#1e1e1e")
status_label.pack(pady=10)

# Email Status Label
email_status_label = tk.Label(root, text="Email Status: None", font=("Helvetica", 18), fg="white", bg="#1e1e1e")
email_status_label.pack(pady=10)

# Start/Stop Buttons
button_frame = tk.Frame(root, bg="#1e1e1e")
button_frame.pack(pady=20)

start_button = tk.Button(button_frame, text="Start Detection", font=("Helvetica", 16), command=lambda: threading.Thread(target=start_detection).start())
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(button_frame, text="Stop Detection", font=("Helvetica", 16), command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=10)

# Run the application
root.mainloop()