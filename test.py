# added by Jincy on June-2nd-2025
from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

app = Flask(__name__)

#  Load label encoder and scaler 
label_encoder = joblib.load("mlp_label_encoder.pkl")
scaler = joblib.load("mlp_scaler.pkl")

#  Define the same model class from training 
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#  Load trained model 
model = MLP(input_size=63, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("gesture_mlp_model.pth", map_location=torch.device("cpu")))
model.eval()

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# routing added for index.html
@app.route("/")
def index():
    return render_template("index.html")

# routing added for asl.html
@app.route("/asl")
def asl():
    return render_template("asl.html")

# routing added for audio.html
@app.route("/audio")
def audio():
    return render_template("audio.html")

# added for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        decoded = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return jsonify({"error": "Invalid image"})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"prediction": "No hand detected", "confidence": 0})

        hand_landmarks = results.multi_hand_landmarks[0]
        base = hand_landmarks.landmark[0]
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])

        # === Preprocess and scale ===
        landmarks_scaled = scaler.transform([landmarks])  # shape: (1, 63)
        input_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32)

        # === Predict ===
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]

        pred_index = np.argmax(probs)
        pred_class = label_encoder.inverse_transform([pred_index])[0]
        confidence = float(probs[pred_index])

        return jsonify({
            "prediction": pred_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
