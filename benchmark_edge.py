import base64
import glob
import json
import mimetypes
import os
import time
import urllib.request
import uuid
from pathlib import Path

API_URL = "http://localhost:8000/predict"
SAMPLES_DIR = "samples"
OUTPUT_DIR = "evidencia_ejecucion"


def send_image_to_api(url: str, image_path: str) -> dict:
    boundary = uuid.uuid4().hex
    filename = os.path.basename(image_path)
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Accept", "application/json")

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    roundtrip_ms = (time.perf_counter() - t0) * 1000.0
    payload["http_roundtrip_ms"] = round(roundtrip_ms, 2)
    return payload


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))
    images = sorted(images)

    if not images:
        print(f"No se encontraron imágenes en la carpeta '{SAMPLES_DIR}/'.")
        return

    # Si solo hay 1 imagen de muestra, se repite 5 veces para medir arranque en frío vs régimen estacionario
    test_queue = images if len(images) >= 3 else images * 5

    print(f"{'Req':<4} | {'Archivo':<20} | {'Pre (ms)':<8} | {'Inf (ms)':<8} | {'Post (ms)':<9} | {'Total (ms)':<10} | {'FPS':<6} | {'Conteo'}")

    latencies = []
    for idx, img_path in enumerate(test_queue, start=1):
        res = send_image_to_api(API_URL, img_path)
        tel = res["telemetry"]
        counts = res["counts"]
        latencies.append(tel["total_latency_ms"])

        counts_str = ", ".join([f"{k}:{v}" for k, v in counts.items() if v > 0]) or "0 piezas"
        print(
            f"{idx:<4} | {os.path.basename(img_path)[:20]:<20} | "
            f"{tel['t_pre_ms']:<8.2f} | {tel['t_inf_ms']:<8.2f} | "
            f"{tel['t_post_ms']:<9.2f} | {tel['total_latency_ms']:<10.2f} | "
            f"{tel['fps']:<6.2f} | {counts_str}"
        )

        # Decodificar y guardar la imagen anotada recibida en Base64 (solo una vez por archivo único)
        if idx <= len(images):
            b64_data = res["image_base64"].split(",", 1)[1]
            out_file = os.path.join(OUTPUT_DIR, f"resultado_{Path(img_path).stem}.jpg")
            with open(out_file, "wb") as out_f:
                out_f.write(base64.b64decode(b64_data))

    steady_latencies = latencies[1:] if len(latencies) > 1 else latencies
    mean_steady = sum(steady_latencies) / len(steady_latencies)
    print(f"Latencia promedio (régimen estacionario): {mean_steady:.2f} ms (~{1000.0 / mean_steady:.2f} FPS)")
    print(f"Imágenes anotadas guardadas en: '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()