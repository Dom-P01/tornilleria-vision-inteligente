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