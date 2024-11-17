import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('model_file.h5')

# Emotion labels (Update this based on your model's labels)
emotions = ['Happy', 'Sad', 'Neutral', 'Angry']  # Modify according to your model

# Preprocess the face to fit the model input
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))  # Resize to 48x48 (adjust if your model uses a different size)
    face = face.astype('float32') / 255.0  # Normalize to [0, 1]
    face = np.expand_dims(face, axis=-1)  # Add channel dimension for grayscale input
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load face detector

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face from the frame
        face = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face)

        # Predict emotion
        prediction = model.predict(processed_face)
        emotion = emotions[np.argmax(prediction)]  # Get the emotion with the highest probability

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the predicted emotion
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the video frame with the bounding box and emotion label
    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
