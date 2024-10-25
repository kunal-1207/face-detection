import cv2
import numpy as np
from deepface import DeepFace

def detect_face(frame):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def analyze_expression_and_age(face_img):
    try:
        # DeepFace analysis for expressions and age
        analysis = DeepFace.analyze(face_img, actions=['emotion', 'age'], enforce_detection=False)
        if analysis:
            return analysis[0]['dominant_emotion'], analysis[0]['age']
    except Exception as e:
        print(f"Error analyzing face: {e}")
    return "Unknown", "Unknown"

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_face(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Analyze facial expression and age
            expression, age = analyze_expression_and_age(face)
            print(f"Facial Expression: {expression}")
            print(f"Estimated Age: {age}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{expression}, Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
