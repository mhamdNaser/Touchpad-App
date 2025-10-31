from sqlalchemy import Column, BigInteger, Integer, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from app.core.database import Base

class Frame(Base):
    __tablename__ = "frames"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    gesture_id = Column(BigInteger, ForeignKey("gestures.id"), nullable=False)
    frame_id = Column(BigInteger, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    points_count = Column(Integer, nullable=False)
    raw_payload = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True)

    gesture = relationship("Gesture", back_populates="frames")
    points = relationship("Point", back_populates="frame")
