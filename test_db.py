from deployment.apps import app, db
from deployment.database.models import User, Patient, XRayCase

with app.app_context():
    db.create_all()
    print("✅ Database connected and tables created")