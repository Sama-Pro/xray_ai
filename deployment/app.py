from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from flask_cors import CORS
import gdown

# -----------------------------
# APP CONFIG
# -----------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

CORS(app)

# -----------------------------
# PORT (FOR RENDER)
# -----------------------------
PORT = int(os.environ.get("PORT", 5000))

# -----------------------------
# MODEL CONFIG
# -----------------------------
FILE_ID = "1KffooXUkliBRSoOiog3PAA-WUtEOHqt2"  # ✅ YOUR CORRECT FILE ID
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.h5")

# -----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (150, 150)
labels = ["NORMAL", "PNEUMONIA"]

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        processed = preprocess_image(img)
        prediction = model.predict(processed)[0][0]

        confidence = float(prediction if prediction > 0.5 else (1 - prediction))
        result = labels[1] if prediction > 0.5 else labels[0]

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=PORT)