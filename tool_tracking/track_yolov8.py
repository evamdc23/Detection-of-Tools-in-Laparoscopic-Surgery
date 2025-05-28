import cv2
import torch
import numpy as np
import yaml
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import sys
from collections import Counter

# Añadir raíz del proyecto al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.bytetrack_wrapper import ByteTrackWrapper

"""
Este script aplica detección de objetos con YOLOv8 y seguimiento multiobjeto con ByteTrack sobre una secuencia 
de imágenes (frames) correspondiente a un video quirúrgico previamente procesado. El objetivo es generar un 
video con los objetos rastreados, así como logs detallados y un resumen cuantitativo de las herramientas detectadas.
"""

# --- CONFIGURACION ---
VIDEO_FOLDER = Path('processed/dataset_for_tracking/frames/vid04')  # Cambia solo esto
video_name = VIDEO_FOLDER.name
MODEL_PATH = Path('results/cholectrack20_yolov8l_mejora1/weights/best.pt')
DATASET_YAML = Path('processed/cholectrack20_for_val/dataset.yaml')
OUTPUT_DIR = Path(f'results_track/tracking_output_validacion_{video_name}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar nombres de clases desde dataset.yaml
with open(DATASET_YAML, 'r') as f:
    class_names = yaml.safe_load(f)['names']

# Mapa de colores fijos por clase
CLASS_COLORS = {
    "grasper": (255, 0, 0),        # azul
    "bipolar": (0, 255, 255),      # amarillo
    "hook": (0, 0, 255),           # rojo
    "scissors": (0, 128, 255),     # naranja
    "clipper": (0, 255, 0),        # verde
    "irrigator": (255, 0, 255),    # rosa
    "specimen_bag": (128, 0, 128)  # morado
}

# Inicializa el modelo YOLO
model = YOLO(str(MODEL_PATH))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inicializa ByteTrack
tracker = ByteTrackWrapper(track_thresh=0.3, buffer_size=90)

# Procesar solo el video dado
print(f"Procesando: {video_name}")
frame_paths = sorted(VIDEO_FOLDER.glob("*.jpg"))
output_video_path = OUTPUT_DIR / f"{video_name}_tracked.mp4"
output_csv_path = OUTPUT_DIR / f"{video_name}_tracks.csv"
output_summary_path = OUTPUT_DIR / f"{video_name}_tools_summary.csv"

writer = None
tracks_log = []
tool_counter = Counter()

for frame_path in frame_paths:
    frame = cv2.imread(str(frame_path))
    h, w, _ = frame.shape

    # Inferencia con YOLOv8
    results = model(frame, device=device)[0]
    detections = results.boxes

    bboxes_xyxy = []
    confidences = []
    class_ids = []

    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        conf = float(det.conf)
        cls = int(det.cls)
        bboxes_xyxy.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(cls)
        tool_counter[class_names.get(cls, str(cls))] += 1

    # Seguimiento con ByteTrack
    outputs = tracker.update(bboxes_xyxy, confidences, class_ids, frame)

    # Visualización y logging
    for i, track in enumerate(outputs):
        x1, y1, x2, y2, track_id, cls = track
        class_name = class_names.get(cls, str(cls))
        conf = confidences[i] if i < len(confidences) else 0.0
        label = f"{class_name} ({conf:.2f})"

        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        tracks_log.append([video_name, frame_path.name, track_id, class_name, conf, x1, y1, x2, y2])

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video_path), fourcc, 1.0, (w, h))

    writer.write(frame)

writer.release()

# Guardar resultados por frame
df = pd.DataFrame(tracks_log, columns=["video", "frame", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
df.to_csv(output_csv_path, index=False)

# Guardar resumen de herramientas detectadas
pd.DataFrame(tool_counter.items(), columns=["tool", "count"]).to_csv(output_summary_path, index=False)

print(f"Resultado guardado en: {output_video_path}")