import cv2
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt

# Suppress TensorFlow OneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the pre-trained model and Haar Cascade classifier
model = load_model('model_file.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define labels for emotion prediction
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize video capture
video = cv2.VideoCapture(0)  # 0 for the default webcam

# Check if the video capture is opened successfully
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    # Process detected faces
    for x, y, w, h in faces:
        # Extract the region of interest (ROI) and preprocess it
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))

        # Predict emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw rectangles and text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame using an alternative method if `cv2.imshow` fails
    try:
        cv2.imshow("Emotion Detection", frame)
    except cv2.error:
        # If `cv2.imshow` fails, use matplotlib for display
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        break

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
