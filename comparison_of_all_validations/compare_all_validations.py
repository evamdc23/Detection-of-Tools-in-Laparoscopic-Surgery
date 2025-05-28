import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

"""
Comparar el desempeño de distintos modelos YOLOv8 (variantes s, l, x, y mejora1) 
usando métricas estándar extraídas desde archivos JSON de validación previamente guardados.
"""

# Rutas a las métricas en JSON
models = {
    "yolov8s": "results_validacion/validacion_yolov8s/metrics_summary.json",
    "yolov8l": "results_validacion/validacion_yolov8l/metrics_summary.json",
    "yolov8x": "results_validacion/validacion_yolov8x/metrics_summary.json",
    "yolov8l_mejora1": "results_validacion/validacion_yolov8l_mejora1/metrics_summary.json",
    "cholectrack20_for_val": "results_validacion/validacion_cholectrack20_for_val/metrics_summary.json" # Este es el unico con otro val que no han visto
}

summary = []

for name, path_str in models.items():
    path = Path(path_str)
    if not path.exists():
        print(f"No encontrado: {path}")
        continue
    with open(path, "r") as f:
        data = json.load(f)
        summary.append({
            "model": name,
            "mAP50": data.get("metrics/mAP50(B)", 0),
            "mAP50-95": data.get("metrics/mAP50-95(B)", 0),
            "precision": data.get("metrics/precision(B)", 0),
            "recall": data.get("metrics/recall(B)", 0),
            "f1": data.get("metrics/f1(B)", 0)
        })

# Crear DataFrame y guardar como CSV
df = pd.DataFrame(summary)
df_sorted = df.sort_values(by="mAP50-95", ascending=False)
print("\n Comparación global de métricas:")
print(df_sorted.to_string(index=False))

output_dir = Path("results_validacion/comparacion")
output_dir.mkdir(parents=True, exist_ok=True)
df_sorted.to_csv(output_dir / "comparacion_modelos.csv", index=False)

# Exportar como tabla Markdown
df_sorted.to_markdown(output_dir / "comparacion_modelos.md", index=False)

# Gráfica de mAP
plt.figure(figsize=(10, 6))
x = df_sorted["model"]
bar_width = 0.35

plt.bar(x, df_sorted["mAP50"], width=bar_width, label="mAP@0.5")
plt.bar(x, df_sorted["mAP50-95"], width=bar_width, label="mAP@0.5:0.95", alpha=0.7)
plt.ylabel("Score")
plt.title("Comparación de modelos - mAP")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "map_comparison.png")

# Gráfica de precisión, recall y f1
plt.figure(figsize=(10, 6))
plt.bar(x, df_sorted["precision"], width=bar_width, label="Precision")
plt.bar(x, df_sorted["recall"], width=bar_width, label="Recall", alpha=0.7)
plt.bar(x, df_sorted["f1"], width=bar_width, label="F1-score", alpha=0.7)
plt.ylabel("Score")
plt.title("Comparación de modelos - Precision, Recall y F1-score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "precision_recall_f1_comparison.png")