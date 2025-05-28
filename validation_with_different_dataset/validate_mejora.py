from ultralytics import YOLO
from pathlib import Path
import shutil
import json

"""
Este script realiza una evaluación completa del modelo YOLOv8 entrenado para el dataset cholectrack20, 
y genera resultados de validación reproducibles. 
Este script se utiliza después del entrenamiento, para verificar cuantitativamente la calidad del 
modelo antes de realizar inferencias masivas o exportarlo a producción.
"""

# Cargar el modelo
model = YOLO('results/cholectrack20_yolov8l_mejora1/weights/best.pt')

# Directorio de salida para resultados de validación
output_dir = Path("results_validacion/validacion_cholectrack20_for_val")
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Ejecutar la validación y guardar los resultados
metrics = model.val(
    data='processed/cholectrack20_for_val/dataset.yaml', # Cambiar
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(output_dir.parent),
    name=output_dir.name,
    exist_ok=True
)

# Guardar las métricas en un archivo JSON
with open(output_dir / 'metrics_summary.json', 'w') as f:
    json.dump(metrics.results_dict, f, indent=4)

print("\n Validación completada. Resultados guardados en:", output_dir.resolve())