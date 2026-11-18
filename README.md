# API de Inspección Visual de Tornillería Industrial (YOLOv8 + FastAPI)

Microservicio REST autocontenido para la detección, clasificación y conteo automatizado de piezas de fijación mecánica (`nuts`, `screws`, `washers`) en líneas de inspección industrial y entornos *edge*.

## Arquitectura y Especificaciones del Modelo

El sistema desacopla la lógica de visión por computadora (`IndustrialInferencePipeline`) de la capa de transporte HTTP (`FastAPI`), operando íntegramente en memoria RAM mediante decodificación directa de flujos binarios (`cv2.imdecode`)[cite: 1].

* **Arquitectura neuronal:** YOLOv8n (*Nano*, diseño *anchor-free*).
* **Resolución de entrada:** 640 x 640 píxeles.
* **Clases soportadas:** `0: nuts` (tuercas), `1: screws` (tornillos), `2: washers` (arandelas).
* **Umbrales operativos por defecto:** Confianza $\tau_{\text{conf}} = 0.40$ e intersección sobre unión *IoU = 0.50*.
* **Desempeño validado:** mAP50 medio de $0.8814 \pm 0.0563$ en validación cruzada de 5 pliegues y latencia en régimen estacionario de 25.53 ms (~39.28 FPS) en GPU T4.

## Estructura del Paquete de Entrega

```text
tornilleria-vision-api/
├── models/
│   └── best.pt               # Pesos entrenados del detector YOLOv8n (~6 MB)
├── samples/
│   └── muestra_test.jpg      # Imagen de muestra para pruebas funcionales[cite: 1]
├── src/
│   ├── __init__.py           # Inicializador del paquete
│   ├── pipeline.py           # Preprocesamiento, inferencia, postprocesamiento y Base64[cite: 1]
│   └── app.py                # Endpoints REST y validación de peticiones con FastAPI[cite: 1]
├── .dockerignore             # Exclusiones del contexto de compilación Docker[cite: 1]
├── .gitignore                # Exclusiones de control de versiones Git[cite: 1]
├── Dockerfile                # Definición de imagen de producción optimizada para CPU/Edge[cite: 1]
├── README.md                 # Documentación técnica del sistema[cite: 1]
└── requirements.txt          # Dependencias ligeras de Python[cite: 1]
```

## Requisitos del sistema

* **Contenedorización:** Docker Engine 24.0+ o Docker Desktop (compatible con arquitecturas `amd64` y `arm64` / Apple Silicon).
* **Hardware mínimo recomendado (Edge/CPU):** 2 vCPU, 2 GB de memoria RAM libre y 3 GB de almacenamiento disponible.

## Construcción y Despliegue con Docker

### Construir la imagen del contenedor
Desde la raíz del repositorio, ejecuta:

```bash
docker build -t tornilleria-vision-api:1.0.0 .
```

### Ejecutar el contenedor en modo servicio
Levanta el microservicio exponiendo el puerto `8000` del host:

```bash
docker run -d \
  --name api-tornilleria \
  -p 8000:8000 \
  -e CONF_THRESHOLD=0.40 \
  -e IOU_THRESHOLD=0.50 \
  tornilleria-vision-api:1.0.0
```

### Variables de entorno configurables

| Variable | Valor por defecto | Descripción |
| :--- | :--- | :--- |
| `MODEL_PATH` | `models/best.pt` | Ruta interna al archivo de pesos PyTorch del modelo. |
| `CONF_THRESHOLD` | `0.40` | Umbral mínimo de confianza para aceptar una detección. |
| `IOU_THRESHOLD` | `0.50` | Umbral de solapamiento para Supresión de No Máximos (NMS). |

## Uso de la API y Contrato de respuesta

Una vez iniciado el contenedor, la documentación interactiva (Swagger UI) queda disponible en:
`http://localhost:8000/docs`[cite: 1]

### Verificar estado del servicio (`GET /health`)
```bash
curl http://localhost:8000/health
```

### Ejecutar inferencia sobre una imagen (`POST /predict`)
Envía una imagen (`JPG`, `PNG`, `BMP`) mediante `multipart/form-data`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@samples/muestra_test.jpg"
```

### Estructura de respuesta JSON (`200 OK`)
El endpoint retorna la telemetría de ejecución en milisegundos, el conteo consolidado por categoría, las coordenadas de cada caja delimitadora y la imagen anotada codificada en Base64

```json
{
  "status": "success",
  "telemetry": {
    "t_pre_ms": 1.4,
    "t_inf_ms": 21.3,
    "t_post_ms": 2.1,
    "total_latency_ms": 24.8,
    "fps": 40.3
  },
  "counts": {
    "nuts": 0,
    "screws": 0,
    "washers": 6
  },
  "detections": [
    {
      "class": "washers",
      "confidence": 0.9412,
      "box_xyxy": [142.5, 88.0, 210.3, 155.8]
    }
  ],
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```
## Resultados de Despliegue y Telemetría Edge (CPU)

Para validar la viabilidad del microservicio en entornos sin aceleración gráfica dedicada, se ejecutó una batería de pruebas funcionales y de estrés sobre el contenedor `api-tornilleria` (entorno Linux de 2 vCPU y 7.76 GiB de RAM) mediante el script `benchmark_edge.py`.

### Huella de recursos del contenedor (`docker stats`)

| Indicador | Medición registrada | Observación operativa |
| :--- | :--- | :--- |
| **Memoria RAM utilizada** | `377.5 MiB / 7.756 GiB` (`4.75%`) | Huella ligera compatible con dispositivos *edge* de 1–2 GB de RAM. |
| **Uso de CPU (en reposo)** | `0.12%` | Sin consumo residual cuando el servidor Uvicorn espera peticiones. |
| **Procesos activos (`PIDS`)** | `8` | Hilos de ejecución del servidor ASGI y del entorno Python. |
| **Tráfico de red (`NET I/O`)** | `303 kB` entrada / `639 kB` salida | Transferencia de 7 imágenes de entrada y respuestas JSON con Base64. |
| **Lectura/Escritura (`BLOCK I/O`)** | `695 MB` lectura / `4.1 kB` escritura | Carga inicial de librerías y pesos en memoria sin escritura temporal en disco. |

### 6.2 Latencia por Etapa y Rendimiento (`benchmark_edge.py`)

Se procesaron 7 peticiones secuenciales enviando fotografías reales del conjunto de prueba (`samples/`):

| Petición | Archivo de muestra | Preproceso (ms) | Inferencia (ms) | Postproceso (ms) | Total (ms) | Rendimiento (FPS) | Piezas detectadas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 (Frío)** | `flat_metal_washers_h...` | 13.10 | 2907.96 | 93.70 | 3014.76 | 0.33 | `washers: 19` |
| **2** | `flat_metal_washers_h...` | 0.06 | 146.69 | 1.32 | 148.08 | 6.75 | `washers: 2` |
| **3** | `metal_screws_hardwar...` | 0.06 | 139.92 | 6.87 | 146.85 | 6.81 | `nuts: 9, screws: 43` |
| **4** | `metal_screws_hardwar...` | 0.07 | 143.30 | 2.26 | 145.63 | 6.87 | `screws: 13` |
| **5** | `screws_and_nuts_scat...` | 0.05 | 149.09 | 3.78 | 152.93 | 6.54 | `nuts: 1, screws: 20` |
| **6** | `steel_hex_nuts_hardw...` | 0.07 | 138.58 | 1.34 | 139.99 | 7.14 | `nuts: 1, washers: 1` |
| **7** | `steel_hex_nuts_hardw...` | 0.05 | 160.51 | 3.03 | 163.60 | 6.11 | `nuts: 26` |
| **Promedio (2–7)** | **Régimen estacionario** | **0.06** | **146.35** | **3.10** | **149.51** | **~6.69** | **138 piezas en 6 imgs** |

* **Arranque en frío (*Cold Start*):** La primera petición registró `3014.76 ms` debido a la inicialización de grafos de PyTorch en CPU. A partir de la segunda petición, el sistema se estabilizó en `149.51 ms` por imagen.
* **Eficiencia en memoria:** El preprocesamiento (`cv2.imdecode` y conversión a RGB) consume en promedio `0.06 ms`, mientras que el postprocesamiento (NMS, dibujo de cajas y codificación JPEG Base64) oscila entre `1.32 ms` y `6.87 ms` según la densidad de piezas.

### Comparativa de Entornos: GPU Cloud vs. CPU Edge

| Métrica operativa | Desarrollo en GPU (Tesla T4) | Despliegue Contenedorizado (CPU 2 vCPU) |
| :--- | :---: | :---: |
| **Latencia total promedio** | `25.53 ms` | `149.51 ms` |
| **Tasa de procesamiento** | `39.28 FPS` | `~6.69 FPS` |
| **Consumo de memoria RAM** | Entorno compartido (~12 GB) | `377.5 MiB` aislados |
| **Aplicación recomendada** | Flujo de video continuo (30 FPS) | Estaciones de inspección por disparo y APIs web sin GPU |

* **Reproducibilidad de la prueba:** Para volver a ejecutar esta medición con el contenedor activo en el puerto `8000`, ejecuta:
  ```bash
  python3 benchmark_edge.py
  docker stats api-tornilleria --no-stream
  ```
