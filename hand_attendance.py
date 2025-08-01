import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import distance
from datetime import datetime

# === Setup ===
dataset_path = "hand_dataset"
log_file = "attendance_log.csv"
threshold = 0.2  # Distance threshold for matching

# Load all student data from dataset
def load_student_data():
    students = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(dataset_path, file))
            if not df.empty:
                students[name] = df.values
    return students

# Compare new hand landmarks to each student's dataset
def recognize_hand(landmarks, students_data):
    for name, samples in students_data.items():
        for sample in samples:
            dist = distance.euclidean(landmarks, sample)
            if dist < threshold:
                return name
    return None

# Mark attendance in a CSV file with timestamp
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if not ((log_df["Name"] == name) & (log_df["Date"] == date_str)).any():
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
        log_df.to_csv(log_file, index=False)
        print(f"[INFO] Attendance marked for {name} at {time_str}")
        return True
    else:
        print(f"[INFO] {name} already marked today.")
        return False

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load data
students_data = load_student_data()
print(f"[INFO] Loaded data for: {list(students_data.keys())}")

# Camera
cap = cv2.VideoCapture(0)
print("[INFO] Show your hand... Press ESC to exit")

marked_today = set()
feedback_message = ""
feedback_color = (0, 255, 0)
feedback_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = "Unrecognized"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
            if len(landmarks) == 42:
                match = recognize_hand(landmarks, students_data)
                if match:
                    label = f"Hello, {match}"
                    if match not in marked_today:
                        if mark_attendance(match):
                            marked_today.add(match)
                            feedback_message = f"✔ Attendance marked for {match}"
                            feedback_color = (0, 255, 0)
                            feedback_time = cv2.getTickCount()
                    else:
                        feedback_message = f"⚠ {match} already marked today"
                        feedback_color = (0, 255, 255)
                        feedback_time = cv2.getTickCount()

    # Draw main label
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if label != "Unrecognized" else (0, 0, 255), 2)

    # Draw popup message
    if feedback_message and (cv2.getTickCount() - feedback_time) / cv2.getTickFrequency() < 2:
        cv2.putText(frame, feedback_message, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)
    else:
        feedback_message = ""

    cv2.imshow("Hand Recognition - Attendance", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
