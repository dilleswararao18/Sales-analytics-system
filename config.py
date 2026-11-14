
import os
from dotenv import load_dotenv
from sqlalchemy import URL

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    SQLALCHEMY_DATABASE_URI = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "sales_analytics")
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
