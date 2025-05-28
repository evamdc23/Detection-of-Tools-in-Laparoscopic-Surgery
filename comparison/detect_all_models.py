import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

"""
Este script ejecuta inferencia con múltiples modelos YOLOv8 sobre un conjunto de imágenes organizadas por nivel de 
dificultad (fáciles y difíciles). Aplica cada modelo sobre ambos subconjuntos y guarda los resultados visuales y 
textuales de las predicciones.
"""

# Modelos a aplicar
models = {
    "yolov8s": "results/cholectrack20_yolov8s/weights/best.pt",
    "yolov8l": "results/cholectrack20_yolov8l/weights/best.pt",
    "yolov8x": "results/cholectrack20_yolov8x/weights/best.pt",
    "yolov8l_mejora1": "results/cholectrack20_yolov8l_mejora1/weights/best.pt" # Este se añade despues (en el paso 9)
}

# Categorías de dificultad
categories = ["faciles", "dificiles"]

# Input base
input_base = Path("processed/test_inference")
output_base = Path("results_detect")

for model_name, model_path in models.items():
    print(f"\nEjecutando modelo: {model_name}")
    model = YOLO(model_path)

    for cat in categories:
        input_path = input_base / cat
        output_dir = output_base / model_name / cat

        # Ejecutar inferencia
        results = model.predict(
            source=str(input_path),
            save=True,
            save_txt=True,
            save_conf=True,
            device=0,
            imgsz=1024,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True
        )

        print(f"Guardado en: {output_dir}")
