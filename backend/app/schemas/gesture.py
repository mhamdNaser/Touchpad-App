from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float

class Frame(BaseModel):
    ts: int 
    frame_id: int
    points: List[Point]

class GestureData(BaseModel):
    start_time: int
    end_time: int
    duration_ms: int
    frame_count: int
    frames: List[Frame]
