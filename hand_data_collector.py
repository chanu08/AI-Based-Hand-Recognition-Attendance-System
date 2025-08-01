import os
import cv2
import mediapipe as mp
import csv

# === Step 1: Setup ===
# Create folder for storing hand landmark data (one file per student)
os.makedirs("hand_dataset", exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Ask user for student name
student_name = input("Enter student name (no spaces): ").strip().lower()
student_file = f"hand_dataset/{student_name}.csv"

# Create/open the CSV file to write landmark data
csvfile = open(student_file, mode="w", newline="")
csvwriter = csv.writer(csvfile)
csvwriter.writerow([f"{i}_{axis}" for i in range(21) for axis in ("x", "y")])  # Header

print(f"[INFO] Collecting data for: {student_name}")
print("[INFO] Place hand inside the box... Press ESC to stop.")

count = 0
max_samples = 300  # Number of hand samples to capture

# Main capture loop
while cap.isOpened() and count < max_samples:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks for feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmark_row = []
            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y])  # z can be added optionally

            # Save to CSV
            csvwriter.writerow(landmark_row)
            count += 1
            print(f"[INFO] Captured sample {count}/{max_samples}")

    # Display frame
    cv2.putText(frame, f"Capturing for: {student_name} ({count}/{max_samples})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Hand Data Collection", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
        break

# Cleanup
csvfile.close()
cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Data collection completed. {count} samples saved to: {student_file}")
