from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import joblib
import numpy as np

app = Flask(__name__)

# === Load encoder, scaler, and model ===
label_encoder = joblib.load("mlp_label_encoder.pkl")
scaler = joblib.load("mlp_scaler.pkl")

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

model = MLP(input_size=63, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("gesture_mlp_model.pth", map_location=torch.device("cpu")))
model.eval()

# === ROUTES ===

@app.route("/")
def splash1():
    return render_template("splash1.html")

@app.route("/splash2")
def splash2():
    return render_template("splash2.html")

@app.route("/splash3")
def splash3():
    return render_template("splash3.html")

@app.route("/splash-carousel")
def splash_carousel():
    return render_template("splash_carousel.html")

@app.route("/userselect")
def user_select():
    return render_template("userSelect.html")

@app.route("/asl")
def asl():
    return render_template("asl.html")

@app.route("/audio")
def audio():
    return render_template("audio.html")

@app.route("/accessibility")
def accessibility():
    return render_template("accessibility.html")

# === LANDMARK-BASED PREDICTION ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("data received:", data)
        landmarks = data.get("landmarks")
        print("Landmark length:", len(landmarks) if landmarks else "None")
        if not landmarks or len(landmarks) != 63:
            return jsonify({"prediction": "Invalid landmark data", "confidence": 0})

        scaled = scaler.transform([landmarks])
        input_tensor = torch.tensor(scaled, dtype=torch.float32)

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
