from pydantic import BaseModel
from fastapi import UploadFile, Form, File
from typing import List

class InputModel(BaseModel):
    name: str = Form(...)


