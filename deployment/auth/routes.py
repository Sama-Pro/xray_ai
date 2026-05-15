from flask import Blueprint, request, jsonify, session
from deployment.apps import db
from deployment.database.models import User, Hospital
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__)

# -------------------------
# REGISTER
# -------------------------
@auth_bp.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "doctor")
    hospital_id = data.get("hospital_id")

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists"}), 400

    user = User(
        name=name,
        email=email,
        role=role,
        hospital_id=hospital_id
    )

    user.password_hash = generate_password_hash(password)

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"})


# -------------------------
# LOGIN
# -------------------------
@auth_bp.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401

    # SESSION STORAGE
    session["user_id"] = user.id
    session["role"] = user.role
    session["hospital_id"] = user.hospital_id

    return jsonify({
        "message": "Login successful",
        "role": user.role
    })


# -------------------------
# LOGOUT
# -------------------------
@auth_bp.route("/logout")
def logout():

    session.clear()

    return jsonify({"message": "Logged out successfully"})


# -------------------------
# CURRENT USER
# -------------------------
@auth_bp.route("/me")
def me():

    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    user = User.query.get(user_id)

    return jsonify({
        "name": user.name,
        "email": user.email,
        "role": user.role
    })