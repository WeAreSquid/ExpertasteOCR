from pydantic import BaseModel
from typing import List

class InputModel(BaseModel):
    name: str
    cropped_card: str
    points: List[List[float]]
    center: List[float]
    array_shape: List[int]



