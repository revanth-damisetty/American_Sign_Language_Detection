from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# --- Load Model and Preprocessors from 'TiltedModel' Folder ---
MODEL_DIR = "TiltedModel"
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "sign_model_3dtilt.h5"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# --- MediaPipe Hands Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# --- Landmark Extraction ---
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    for hand_landmarks in results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None

# --- Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    landmarks = extract_landmarks(image)
    if landmarks is None:
        return jsonify({'error': 'No hand detected'}), 400

    landmarks = scaler.transform([landmarks])
    prediction = model.predict(landmarks)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    return jsonify({
        'sign': predicted_label,
        'confidence': confidence
    })

# --- Run Server ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
