from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from flask_cors import CORS
import gdown
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Grad-CAM
from deployment.gradcam import make_gradcam_heatmap, save_gradcam

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
FILE_ID = "1z-sfKqHSbuhvsGhTJLLTYP_8EF12Jv8A"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_models.h5"
)

# -----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -----------------------------
if not os.path.exists(MODEL_PATH):

    print("📥 Downloading model from Google Drive...")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

# -----------------------------
# LOAD MODEL
# -----------------------------
print("📦 Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (224, 224)

labels = ["NORMAL", "PNEUMONIA"]

THRESHOLD = 0.60

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(image):

    image = image.resize(IMG_SIZE)

    image = np.array(
        image,
        dtype=np.float32
    )

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

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

        # -----------------------------
        # CHECK IMAGE
        # -----------------------------
        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        # -----------------------------
        # SAVE IMAGE
        # -----------------------------
        upload_folder = os.path.join(
            BASE_DIR,
            "static",
            "uploads"
        )

        os.makedirs(
            upload_folder,
            exist_ok=True
        )

        filepath = os.path.join(
            upload_folder,
            file.filename
        )

        file.save(filepath)

        # -----------------------------
        # OPEN IMAGE
        # -----------------------------
        img = Image.open(
            filepath
        ).convert("RGB")

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        processed = preprocess_image(img)

        # -----------------------------
        # PREDICT
        # -----------------------------
        prediction = model.predict(processed)[0][0]

        # -----------------------------
        # RESULT
        # -----------------------------
        result = (
            labels[1]
            if prediction > THRESHOLD
            else labels[0]
        )

        # -----------------------------
        # CONFIDENCE
        # -----------------------------
        confidence = float(
            max(
                prediction,
                1 - prediction
            ) * 100
        )

        # -----------------------------
        # GRAD-CAM
        # -----------------------------
        last_conv_layer_name = "block_16_project"

        heatmap = make_gradcam_heatmap(
            processed,
            model,
            last_conv_layer_name
        )

        heatmap_path = os.path.join(
            upload_folder,
            "gradcam.jpg"
        )

        save_gradcam(
            filepath,
            heatmap,
            heatmap_path
        )

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return jsonify({

            "prediction": result,

            "confidence": round(
                confidence,
                2
            ),

            "gradcam": "/static/uploads/gradcam.jpg"

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
    

    # -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":

    print("🚀 Starting Flask server...")

    app.run(
        host="0.0.0.0",
        port=PORT
    )