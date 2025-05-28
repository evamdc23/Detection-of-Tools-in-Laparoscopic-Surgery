import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

"""
Este script selecciona aleatoriamente imágenes anotadas de distintos datasets quirúrgicos y 
las clasifica en dos categorías según su nivel de dificultad (basado en la cantidad de objetos 
anotados por imagen). Las imágenes seleccionadas se copian a una estructura organizada para 
tareas de inferencias de prueba o validación cualitativa.
"""

# Configuración general
output_root = Path("processed/test_inference")
folders = {
    "cholec80": Path("processed/cholec80"),
    "m2cai16": Path("processed/m2cai16"),
    "surgtoolloc": Path("processed/surgtoolloc")
}

# Crear carpetas de salida
for difficulty in ["faciles", "dificiles"]:
    (output_root / difficulty).mkdir(parents=True, exist_ok=True)

# Cuántas queremos por dataset y dificultad
MAX_PER_CATEGORY = 50
random.seed(42)  # Para reproducibilidad

# Recoger archivos clasificados por dataset y dificultad
images_by_category = defaultdict(lambda: {"faciles": [], "dificiles": []})

# Recorremos cada dataset
for dataset_name, base_path in folders.items():
    labels_path = base_path / "labels"
    frames_path = base_path / "frames"

    for root, _, files in os.walk(labels_path):
        for file in files:
            if not file.endswith(".txt"):
                continue

            label_file = Path(root) / file
            with open(label_file, "r") as f:
                lines = f.readlines()

            num_objects = len(lines)

            if num_objects == 0:
                continue  # ignoramos imágenes sin herramientas
            elif num_objects == 1:
                difficulty = "faciles"
            else:
                difficulty = "dificiles"

            # Construir ruta de la imagen
            rel_path = label_file.relative_to(labels_path).with_suffix(".jpg")
            image_file = frames_path / rel_path

            if image_file.exists():
                images_by_category[dataset_name][difficulty].append((image_file, rel_path.stem))

# Seleccionar aleatoriamente y copiar
for dataset_name, difficulties in images_by_category.items():
    for difficulty, items in difficulties.items():
        selected = random.sample(items, min(MAX_PER_CATEGORY, len(items)))
        for image_path, stem in selected:
            new_name = f"{dataset_name}_{difficulty}_{stem}.jpg"
            dest = output_root / difficulty / new_name
            shutil.copy(image_path, dest)

print("\nProceso aleatorio completado. Resultado en:", output_root.absolute())
print("Resumen:")
for ds, categories in images_by_category.items():
    for diff in ["faciles", "dificiles"]:
        print(f"  {ds} - {diff}: {min(MAX_PER_CATEGORY, len(categories[diff]))} seleccionadas")
