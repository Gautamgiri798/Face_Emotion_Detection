# Face_Emotion_Detection

Face Emotion Detection using Python, OpenCV, and Keras
📌 Overview
This project is a Real-Time Face Emotion Detection System using Python, OpenCV, and Keras. It utilizes a pre-trained deep learning model to recognize human emotions from facial expressions captured via a webcam.

🛠️ Features
Real-time face detection using OpenCV.
Emotion classification using a pre-trained deep learning model.
Supports emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
Adjustable screen size and resolution for better user experience.

1.Install Dependencies
Make sure you have Python 3.7+ installed.
pip install -r requirements.txt

2.Download the Model Files
Ensure you have the following model files in your project directory:
emotiondetector.json (Model architecture)
emotiondetector.h5 (Trained weights)
If not, train a new model or download it from a pre-trained source.

🎯 Usage
Run the script to start real-time emotion detection.
python face_emotion_detection.py

📷 How It Works
Webcam Captures Frame → The program captures a frame from your webcam.
Face Detection → OpenCV’s Haar Cascade detects faces in the frame.
Emotion Classification → The face region is extracted and passed to the deep learning model for emotion classification.
Display Prediction → The predicted emotion is displayed on the screen with a bounding box.

Controls:
Press 'Q' to exit the webcam window.

📜 License
This project is licensed under the MIT License.

📧 Contact
For any inquiries, contact: 📧 Email: your-gautamgiri672@gmail.com 

