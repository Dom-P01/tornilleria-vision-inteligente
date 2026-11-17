import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.pipeline import IndustrialInferencePipeline

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.40"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.50"))

pipeline_instance: IndustrialInferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    pipeline_instance = IndustrialInferencePipeline(
        weights_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    yield
    pipeline_instance = None


app = FastAPI(
    title="API de Inspección de Tornillería Industrial",
    description="Microservicio de detección y conteo de piezas (nuts, screws, washers) con YOLOv8n.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline_instance is not None,
        "weights_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="El modelo de inferencia no está inicializado.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado debe ser una imagen válida.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen enviada.")

    try:
        result_payload = pipeline_instance.run_inference(img_bgr)
        return result_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno durante la inferencia: {exc}") from exc