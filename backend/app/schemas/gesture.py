from pydantic import BaseModel
from typing import List

# Touchpad readings
class GestureCreate(BaseModel):
    data: List[float] 
