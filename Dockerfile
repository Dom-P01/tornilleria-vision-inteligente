# Imagen base ligera de Python (soporte multi-arquitectura amd64/arm64)
FROM python:3.11-slim

# Evitar archivos .pyc y habilitar salida directa de logs en consola
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=models/best.pt \
    CONF_THRESHOLD=0.40 \
    IOU_THRESHOLD=0.50

WORKDIR /app

# Librerías del sistema requeridas por OpenCV en Linux slim (resuelve libxcb.so.1 y libGL.so.1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libx11-6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Instalación de PyTorch en variante CPU para reducir el peso de la imagen (~2 GB menos)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalación de dependencias de producción del microservicio
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente modular y pesos entrenados del modelo
COPY src/ ./src/
COPY models/ ./models/

# Creación de usuario sin privilegios de administrador por seguridad operativa
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer puerto del servicio REST
EXPOSE 8000

# Verificación automática de salud del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de arranque del servidor ASGI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]