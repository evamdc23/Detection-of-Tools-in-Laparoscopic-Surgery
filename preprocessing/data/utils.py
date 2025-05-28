# Funciones compartidas
import cv2
import numpy as np
from config.paths import paths
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Este módulo contiene funciones utilitarias compartidas que ayudan en la validación de datos, 
la generación de anotaciones en formato YOLO, y la visualización gráfica de etiquetas sobre imágenes. 
Estas funciones están diseñadas para apoyar el desarrollo y depuración de modelos de detección de herramientas quirúrgicas.
"""

# VALIDACIONES DE DATOS

def validate_video(video_path: Path) -> bool:
    # Abre el vídeo y comprueba que se pueda leer y que tenga al menos un frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length > 0

def validate_image_sequence(video_id: str) -> bool:
    frame_folder = paths.DATASETS["cholect50"]["videos"] / video_id
    return frame_folder.exists() and any(frame_folder.glob("*.png"))

# GENERACIÓN DE ANOTACIONES

def generate_yolo_annotation(output_path: Path, class_id: int, presence: int):
    # Genera un archivo de anotación YOLO con una caja que ocupa toda la imagen si presence=1.
    # Se usa para datasets que no tienen bounding boxes, solo presencia binaria.
    if presence:
        with open(output_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n") # x_center, y_center, width, height

# VISUALIZACIÓN DE ANOTACIONES

def visualize_annotations(image_path, label_path, class_names, box_format="xywh"):
    # Muestra una imagen con los bounding boxes en formato YOLO (clase cx cy w h).
    # Ideal para verificar visualmente las etiquetas generadas.
    image_path = Path(image_path)
    label_path = Path(label_path)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img)

    # Colores por clase
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                cx *= w
                cy *= h
                bw *= w
                bh *= h
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                color = colors[int(cls_id) % len(colors)]
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, class_names[int(cls_id)], color=color, fontsize=8, weight='bold')

    # Guardar imagen en carpeta visualizations/
    output_path = Path("visualizations")
    output_path.mkdir(exist_ok=True)
    out_file = output_path / f"viz_{image_path.stem}.jpg"
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Guardado en: {out_file}")