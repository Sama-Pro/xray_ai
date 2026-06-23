
import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")

    if SQLALCHEMY_DATABASE_URI:
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False