import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO

"""
Este script configura y ejecuta el entrenamiento de un modelo YOLOv8 utilizando la biblioteca ultralytics. 
El entrenamiento se basa en un conjunto de datos quirúrgico anotado previamente (en este caso, cholectrack20). 
Utiliza un modelo preentrenado como punto de partida y aplica múltiples técnicas de augmentación de datos.
"""

# Comprobar CUDA
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Cargar modelo preentrenado
model = YOLO('results/cholectrack20_yolov8l/weights/best.pt')  # Cambiar a 'yolov8s.pt', 'yolov8l.pt' o 'yolov8x.pt' para el primer entrenamiento, despues utilizar el mejor modelo

# Entrenamiento
model.train(
    data='../processed/cholectrack20_split/dataset.yaml',
    epochs=180,
    imgsz=1024,
    batch=4,
    patience=20,
    save=True,
    device=0,
    workers=8,
    project='results',
    name='cholectrack20_yolov8l_mejora1',
    exist_ok=True,
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    cos_lr=True,
    warmup_epochs=3,
    close_mosaic=10,
    augment=True,
    degrees=0.2,
    translate=0.1,
    scale=0.1,
    shear=0.1,
    perspective=0.001,
    flipud=0.8,
    fliplr=0.8,
    copy_paste=0.3
)
