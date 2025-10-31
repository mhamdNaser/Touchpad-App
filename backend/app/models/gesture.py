from sqlalchemy import Column, BigInteger, String, Integer, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from app.core.database import Base

class Gesture(Base):
    __tablename__ = "gestures"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    character = Column(String(255), nullable=False)
    user_id = Column(BigInteger, nullable=True)
    device_id = Column(String(255), nullable=True)
    start_time = Column(TIMESTAMP, nullable=True)
    end_time = Column(TIMESTAMP, nullable=True)
    duration_ms = Column(Integer, nullable=False)
    frame_count = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True)

    frames = relationship("Frame", back_populates="gesture")
