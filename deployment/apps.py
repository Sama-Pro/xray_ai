# deployment/apps.py

from flask import Flask, request, jsonify, render_template, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from flask_cors import CORS
import gdown
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime

# Grad-CAM
from deployment.gradcam import make_gradcam_heatmap, save_gradcam

# Config + DB
from deployment.config import Config
from deployment.extensions import db, login_manager

# Models
from deployment.database.models import Patient, XRayCase, User
from deployment.ai_service import run_xray_pipeline


# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.config.from_object(Config)

CORS(app)

db.init_app(app)
login_manager.init_app(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -----------------------------
# CONFIG
# -----------------------------
PORT = int(os.environ.get("PORT", 5000))

FILE_ID = "1z-sfKqHSbuhvsGhTJLLTYP_8EF12Jv8A"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_models.h5")

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# -----------------------------
# MODEL (LAZY LOADING)
# -----------------------------
model = None


def get_model():
    global model

    if model is None:

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # Download if missing
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

    return model


# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (224, 224)
labels = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.60


# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
def preprocess_image(image):

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# REGISTER
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            return "User already exists"

        hashed_password = generate_password_hash(password)

        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        return "Registration successful"

    return render_template("register.html")


# -----------------------------
# LOGIN
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user_id"] = user.id
            session["username"] = user.username

            return redirect("/dashboard")

        return "Invalid credentials"

    return render_template("login.html")


# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    total_patients = Patient.query.count()
    total_cases = XRayCase.query.count()

    pneumonia_cases = XRayCase.query.filter_by(prediction="PNEUMONIA").count()
    normal_cases = XRayCase.query.filter_by(prediction="NORMAL").count()

    recent_cases = db.session.query(
        XRayCase,
        Patient
    ).join(
        Patient,
        XRayCase.patient_id == Patient.id
    ).order_by(
        XRayCase.created_at.desc()
    ).limit(10).all()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_cases=total_cases,
        pneumonia_cases=pneumonia_cases,
        normal_cases=normal_cases,
        recent_cases=recent_cases
    )


# -----------------------------
# PATIENT DETAILS
# -----------------------------
@app.route("/patient/<int:patient_id>")
def patient_details(patient_id):

    if "user_id" not in session:
        return redirect("/login")

    patient = Patient.query.get_or_404(patient_id)

    cases = XRayCase.query.filter_by(
        patient_id=patient.id
    ).order_by(
        XRayCase.created_at.desc()
    ).all()

    return render_template(
        "patient_details.html",
        patient=patient,
        cases=cases
    )


# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# -----------------------------
# PATIENT FORM + AI + GRAD-CAM
# -----------------------------
@app.route("/patient_form", methods=["GET", "POST"])
def patient_form():

    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        full_name = request.form.get("full_name")
        age = request.form.get("age")
        gender = request.form.get("gender")

        new_patient = Patient(
            full_name=full_name,
            age=age,
            gender=gender
        )

        db.session.add(new_patient)
        db.session.commit()

        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]

        if file.filename == "":
            return "Invalid file", 400

        upload_folder = os.path.join(BASE_DIR, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = f"{new_patient.id}_{timestamp}_{file.filename}"
        filepath = os.path.join(upload_folder, filename)

        file.save(filepath)

        web_image_path = "/static/uploads/" + filename

        # -----------------------------
        # AI PIPELINE
        # -----------------------------
        prediction, confidence = run_xray_pipeline(filepath)

        # -----------------------------
        # GRAD-CAM
        # -----------------------------
        img = Image.open(filepath).convert("RGB")
        processed = preprocess_image(img)

        heatmap = make_gradcam_heatmap(
            processed,
            get_model(),
            "block_16_project"
        )

        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(upload_folder, gradcam_filename)

        save_gradcam(filepath, heatmap, gradcam_path)

        web_gradcam_path = "/static/uploads/" + gradcam_filename

        # -----------------------------
        # SAVE DB
        # -----------------------------
        new_case = XRayCase(
            patient_id=new_patient.id,
            image_path=web_image_path,
            prediction=prediction,
            confidence=confidence,
            
        )

        db.session.add(new_case)
        db.session.commit()

        return "Patient registered successfully"

    return render_template("patient_form.html")


# -----------------------------
# API PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        upload_folder = os.path.join(BASE_DIR, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        filename = file.filename
        filepath = os.path.join(upload_folder, filename)

        file.save(filepath)

        web_image_path = "/static/uploads/" + filename

        prediction, confidence = run_xray_pipeline(filepath)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "image": web_image_path,
        
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# START APP
# -----------------------------
if __name__ == "__main__":

    print("🚀 Starting Flask server...")

    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=PORT)