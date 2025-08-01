✋ AI-Based Hand Recognition Attendance System
A real-time, console-based attendance system built with Python, OpenCV, and MediaPipe for AI-powered hand detection. It captures hand landmarks through the webcam and automatically logs attendance to a CSV file — all without using a graphical user interface (GUI).

✅ Features
Real-time webcam detection
AI-based hand recognition using MediaPipe
Auto-marking of attendance with name, date, and time
CSV log file generated daily
Lightweight and runs completely from the terminal

🛠️ Technologies Used
Python 3.x
OpenCV
MediaPipe
NumPy
CSV module

📂 Folder Structure
graphql
Copy
Edit
HandAttendanceSystem/
├── hand_attendance.py         # Main script
├── registered_hands/          # Folder for registered user data (if any)
├── Attendance/                # Daily CSV attendance logs
├── screenshots/               # Optional: screenshots on successful match
└── README.md

🚀 How to Run
Install dependencies:
pip install opencv-python mediapipe pyttsx3

Run the program:
python hand_attendance.py

Show your hand to the webcam.
Attendance will be marked automatically.

📌 Use Cases
College or school classrooms
Labs and research centers
Office or team check-ins
Event or workshop attendance

📈 Future Improvements
Enhanced identity matching
Admin panel for CLI log management
Cloud or database integration
QR or face fallback support

📄 License
This project is licensed under the MIT License.
