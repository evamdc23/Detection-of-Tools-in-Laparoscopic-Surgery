import sys
from pathlib import Path

# Asegura que Python encuentre la carpeta raíz y config/
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from config.paths import paths
from pathlib import Path
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

"""
Este script analiza la distribución de clases de herramientas quirúrgicas en los datos anotados de un dataset procesado previamente. 
Evalúa la cantidad de anotaciones YOLO por clase, detecta posibles desajustes entre frames y etiquetas, e imprime y grafica los resultados.
"""

def analyze_tool_distribution(dataset_name: str):
    labels_base = paths.PROCESSED_LABELS[dataset_name]
    frames_base = paths.PROCESSED_FRAMES[dataset_name]
    class_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag"]

    tool_counts = defaultdict(int)
    total_labels = 0
    video_discrepancies = []

    for video_folder in labels_base.iterdir():
        if not video_folder.is_dir():
            continue

        label_files = list(video_folder.glob("*.txt"))
        total_labels += len(label_files)

        # Verificar cantidad de imágenes
        video_id = video_folder.name
        frame_folder = frames_base / video_id
        image_files = list(frame_folder.glob("*.jpg"))

        if len(image_files) != len(label_files):
            video_discrepancies.append((video_id, len(image_files), len(label_files)))

        for label_file in label_files:
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        tool_counts[class_id] += 1

    print(f"\nTotal frames con anotaciones: {total_labels}")
    print("Distribución de herramientas:")
    for i, name in enumerate(class_names):
        print(f" - {name:15}: {tool_counts[i]}")

    # Comprobar desajustes entre imágenes y etiquetas
    if video_discrepancies:
        print("\nDesajustes encontrados:")
        for video_id, imgs, labels in video_discrepancies:
            print(f" - {video_id}: {imgs} imágenes, {labels} labels")
    else:
        print("\nTodas las imágenes están correctamente etiquetadas.")

    # Graficar distribución
    plt.figure(figsize=(10, 6))
    plt.bar(
        [class_names[i] for i in range(len(class_names))],
        [tool_counts[i] for i in range(len(class_names))],
        color='skyblue'
    )
    plt.title(f"Distribución de herramientas en {dataset_name}")
    plt.ylabel("Cantidad de anotaciones")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Nombre del dataset a analizar.")
    args = parser.parse_args()
    analyze_tool_distribution(args.dataset)
