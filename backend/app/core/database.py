###############
###############
###############
# In this file,
# we prepare and connect to the database,
# or create it if it does not already exist.
###############
###############
###############

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

DB_NAME = os.environ.get("DB_NAME", "sydev")
EXTERNAL_HOST = os.environ.get("DB_HOST")
EXTERNAL_USER = os.environ.get("DB_USER")
EXTERNAL_PASSWORD = os.environ.get("DB_PASSWORD")

if not all([EXTERNAL_HOST, EXTERNAL_USER, EXTERNAL_PASSWORD]):
    print("⚠️ Warning: No environment variables were found for database connection. Please set DB_HOST, DB_USER, and DB_PASSWORD.")


DATABASE_URL = f"mysql+pymysql://{EXTERNAL_USER}:{EXTERNAL_PASSWORD}@{EXTERNAL_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()