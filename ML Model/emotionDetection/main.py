from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the emotion detection model
model = tf.keras.models.load_model('model_file.h5')

# Preprocessing function (modify as per your model's input)
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))  # Example size, adjust for your model
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Emotion labels
emotions = ['Happy', 'Sad', 'Neutral', 'Angry']  # Update based on your model

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face)
        prediction = model.predict(processed_face)
        emotion = emotions[np.argmax(prediction)]

        # Draw bounding box and emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame

# Video streaming generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Access the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)  # Perform emotion detection
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
