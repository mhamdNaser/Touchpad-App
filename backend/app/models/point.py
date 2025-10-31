from sqlalchemy import Column, BigInteger, Integer, Float, Enum, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

class PointState(str, enum.Enum):
    down = "down"
    move = "move"
    up = "up"
    hover = "hover"

class Point(Base):
    __tablename__ = "points"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    frame_id = Column(BigInteger, ForeignKey("frames.id"), nullable=False)
    point_id = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    state = Column(Enum(PointState), nullable=False)
    pressure = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True)

    frame = relationship("Frame", back_populates="points")
