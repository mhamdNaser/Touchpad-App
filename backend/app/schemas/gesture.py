from pydantic import BaseModel
from typing import List

class GestureCreate(BaseModel):
    data: List[float]  # قراءات التاتش باد
