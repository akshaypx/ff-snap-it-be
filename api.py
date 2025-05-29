from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Allow CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = YOLO("best.pt")

# Response schema
class DetectionResponse(BaseModel):
    annotated_image: str
    detected_items: List[str]

def read_image(file: UploadFile) -> np.ndarray:
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    return np.array(image)

def encode_image(img_array: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_array)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image = read_image(file)

    results = model.predict(source=image, conf=0.5)
    annotated = results[0].plot()

    # Get list of detected item names
    item_names = [model.names[int(cls)] for cls in results[0].boxes.cls]

    return {
        "annotated_image": encode_image(annotated),
        "detected_items": item_names,
    }
