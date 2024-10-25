# Real-Time Facial Expression and Age Detection
This project implements real-time face detection and analysis using OpenCV and DeepFace to estimate facial expressions and approximate age from live video input. The system uses Haar Cascade classifiers for face detection and DeepFace for analyzing emotions and age.

# Features
- Face Detection: Utilizes OpenCV’s Haar Cascade Classifier for efficient face detection.
- Emotion and Age Analysis: Applies DeepFace’s deep learning models to detect emotions and estimate age.
- Real-Time Processing: Runs on live webcam input, displaying detection results on each frame.

# Requirements
The project requires the following libraries:
- Python 3.6+
- OpenCV
- DeepFace
- NumPy
Install dependencies using the command:

      pip install opencv-python deepface numpy

# How to Use
1. Clone or fork this repository to your local machine.
2. Ensure all dependencies are installed.
3. Run the main script:

       python face detection.py

4. A video window will open, showing detected faces with the identified expression and estimated age. Press q to exit.

# Code Overview
- detect_face(frame): Detects faces within a given frame.
- analyze_expression_and_age(face_img): Analyzes the facial expression and age of a detected face.
- main(): Captures video input, applies face detection, and displays results in real-time.

# Usage Note
This code uses live video input and requires a functional webcam. The emotion and age estimation models in DeepFace may be computationally intensive, so performance may vary by device.

# License
This project is licensed under the MIT License.








