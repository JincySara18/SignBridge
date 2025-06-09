from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
import cv2
import mediapipe as mp
import joblib
import os
import logging

logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

# === Load KNN model and label encoder ===
model = joblib.load("gesture_knn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/asl")
def asl():
    return render_template("asl.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info("Entered prediction route")
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        decoded = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            logging.info("Received empty or invalid image.");
            return jsonify({"error": "Received empty or invalid image."})

        # === Convert to RGB and use MediaPipe to detect hand ===
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            logging.info("No hand detected");
            return jsonify({"prediction": "No hand detected", "confidence": 0})

        # === Extract landmarks ===
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # === Predict using KNN model ===
        X = np.array(landmarks).reshape(1, -1)
        pred_index = model.predict(X)[0]
        pred_class = label_encoder.inverse_transform([pred_index])[0]

        # === Approximate confidence using nearest neighbor distance ===
        distances, _ = model.kneighbors(X)
        confidence = float(1 - distances[0][0])  # closer = higher confidence

       logging.info("Predicted class: {pred_class}")
       logging.info("Confidence: {confidence:.4f}")

        return jsonify({
            "prediction": pred_class,
            "confidence": confidence
        })

    except Exception as e:
        logging.error("Something went wrong")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
