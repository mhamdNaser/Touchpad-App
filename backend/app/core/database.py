from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import pymysql

DB_NAME = "sydev"
DATABASE_URL = f"mysql+pymysql://root:@localhost/{DB_NAME}"
ROOT_URL = "mysql+pymysql://root:@localhost" 


def create_database_if_not_exists():
    connection = pymysql.connect(host="localhost", user="root", password="")
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    connection.commit()
    cursor.close()
    connection.close()

# ⬇️ قم أولاً بإنشاء القاعدة
create_database_if_not_exists()

# ⬇️ ثم أنشئ الاتصال والجلسة
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()