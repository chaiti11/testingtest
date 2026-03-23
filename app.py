"""
AI Image Detector - Flask Backend (MobileNetV2 Version)

Usage:
    python app.py
    Open http://localhost:5000
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# TensorFlow / Keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'}

# Load trained MobileNetV2 model
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.h5 not found. Run train_model.py first!")

model = load_model(MODEL_PATH)


# ── Helper ─────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "AI Generated" if prediction > 0.5 else "Real Image"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return label, round(confidence * 100, 2), float(prediction * 100)


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))

        label, confidence, ai_prob = predict_image(img)

        # Create preview
        preview = img.copy()
        preview.thumbnail((400, 400))
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=85)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "ai_probability": round(ai_prob, 2),
            "method": "MobileNetV2",
            "preview": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        os.system("python train_model.py")
        return jsonify({
            "success": True,
            "message": "Model retrained successfully using MobileNetV2"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })


@app.route("/stats")
def stats():
    return jsonify({
        "model_name": "MobileNetV2",
        "status": "Active"
    })


# ── Run App ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 AI Image Detector running at http://localhost:5000")
    print("   Model: MobileNetV2 (Custom Trained)\n")
    app.run(debug=True, host="0.0.0.0", port=5000)