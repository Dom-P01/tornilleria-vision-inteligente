import base64
import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO


class IndustrialInferencePipeline:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
    ):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")

        self.model = YOLO(weights_path)
        self.conf_thresh = conf_threshold
        self.iou_thresh = iou_threshold
        self.class_names = self.model.names  # {0: 'nuts', 1: 'screws', 2: 'washers'}

    def preprocess(self, image_input: Union[str, Path, np.ndarray]):
        """Convierte entradas desde ruta o matriz BGR en memoria a formato RGB."""
        t0 = time.perf_counter()

        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"No se pudo leer la imagen desde disco: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("La entrada debe ser una ruta de archivo o un arreglo NumPy.")

        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Formato de canales de imagen no soportado.")

        h, w = img_rgb.shape[:2]
        t_pre = (time.perf_counter() - t0) * 1000.0
        return img_rgb, (h, w), t_pre

    def predict(self, img_rgb: np.ndarray):
        """Ejecuta el pase hacia adelante de YOLOv8 a resolución 640x640."""
        t0 = time.perf_counter()
        results = self.model.predict(
            source=img_rgb,
            imgsz=640,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        t_inf = (time.perf_counter() - t0) * 1000.0
        return results[0], t_inf

    def postprocess(self, yolo_result) -> tuple[list, dict, str, float]:
        """Estructura detecciones, contabiliza clases y codifica la imagen anotada en Base64."""
        t0 = time.perf_counter()
        boxes = yolo_result.boxes
        detections = []
        counts = {name: 0 for name in self.class_names.values()}

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                cls_name = self.class_names.get(cls_id, f"ID_{cls_id}")
                counts[cls_name] = counts.get(cls_name, 0) + 1
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": round(float(conf), 4),
                        "box_xyxy": [round(float(coord), 2) for coord in box],
                    }
                )

        # Renderizado visual en BGR y codificación JPEG directa en memoria
        annotated_bgr = yolo_result.plot()
        success, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            raise RuntimeError("Error al codificar la imagen anotada en formato JPEG.")

        b64_str = base64.b64encode(buffer).decode("utf-8")
        image_base64 = f"data:image/jpeg;base64,{b64_str}"

        t_post = (time.perf_counter() - t0) * 1000.0
        return detections, counts, image_base64, t_post

    def run_inference(self, image_input: Union[str, Path, np.ndarray]) -> dict:
        """Orquesta el flujo completo y construye el contrato de respuesta."""
        img_rgb, _, t_pre = self.preprocess(image_input)
        yolo_result, t_inf = self.predict(img_rgb)
        detections, counts, image_base64, t_post = self.postprocess(yolo_result)

        total_latency = t_pre + t_inf + t_post
        fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        return {
            "status": "success",
            "telemetry": {
                "t_pre_ms": round(t_pre, 2),
                "t_inf_ms": round(t_inf, 2),
                "t_post_ms": round(t_post, 2),
                "total_latency_ms": round(total_latency, 2),
                "fps": round(fps, 2),
            },
            "counts": counts,
            "detections": detections,
            "image_base64": image_base64,
        }