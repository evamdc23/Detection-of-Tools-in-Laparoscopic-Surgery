import sys
from pathlib import Path

# 1. Añadir ruta raíz del proyecto al sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# 2. Importar después de modificar sys.path
import random
import argparse
from utils import visualize_annotations
from config.paths import paths

"""
Este script permite seleccionar y visualizar aleatoriamente imágenes anotadas de un video específico dentro de un 
dataset quirúrgico ya procesado. Se utiliza principalmente para validar visualmente la calidad y consistencia de 
las anotaciones generadas en formato YOLO.
"""

def pick_random_samples(dataset_name: str, video_id: str, n: int = 3):
    # Directorios de imágenes y labels preprocesadas
    frames_dir = paths.get_processed_frames_path(dataset_name, video_id, create_dir=False)
    labels_dir = paths.PROCESSED_LABELS[dataset_name] / video_id

    # Obtener lista de imágenes .jpg
    all_frames = sorted(list(frames_dir.glob("*.jpg")))
    if len(all_frames) == 0:
        print(f"No se encontraron imágenes en {frames_dir}")
        return

    # Elegir muestras aleatorias
    samples = random.sample(all_frames, min(n, len(all_frames)))

    # Lista de nombres de clases para datasets de 7 herramientas
    class_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen bag"]

    # Visualizar cada imagen con sus anotaciones
    for img_path in samples:
        label_path = labels_dir / (img_path.stem + ".txt")
        print(f"Mostrando: {img_path.name}")
        visualize_annotations(img_path, label_path, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Nombre del dataset.")
    parser.add_argument("--video", required=True, help="ID del video, ej: video01 o VID68")
    args = parser.parse_args()

    pick_random_samples(args.dataset, args.video)
