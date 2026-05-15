import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
THRESHOLD = 0.60

# -----------------------------
# SAFE MODEL PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_models.h5"
)

# Load model once
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32")

    img = np.expand_dims(
        img,
        axis=0
    )

    return preprocess_input(img)


def run_xray_pipeline(image_path):

    # 1. preprocess
    img = preprocess_image(image_path)

    # 2. prediction
    prob = model.predict(img)[0][0]

    # 3. threshold
    if prob > THRESHOLD:
        prediction = "PNEUMONIA"
    else:
        prediction = "NORMAL"

    # 4. confidence %
    confidence = float(
        (prob if prob > THRESHOLD else 1 - prob) * 100
    )

    return prediction, round(confidence, 2)