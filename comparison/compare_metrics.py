import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

"""
Este script analiza y compara los resultados de detección de múltiples modelos YOLOv8 
aplicados a imágenes quirúrgicas clasificadas por dificultad. Extrae métricas clave, 
calcula scores ponderados, genera rankings comparativos, y guarda tanto los resultados 
en CSV como gráficos visuales.
"""

# Directorio raíz de resultados de detección
base_dir = Path("results_detect")
models = ["yolov8s", "yolov8l", "yolov8x", "yolov8l_mejora1"]
categories = ["faciles", "dificiles"]

class_names = {
    0: "grasper", 1: "bipolar", 2: "hook", 3: "scissors",
    4: "clipper", 5: "irrigator", 6: "specimen_bag"
}

# Estructura para almacenar los resultados
summary = []

for model in models:
    for category in categories:
        label_dir = base_dir / model / category / "labels"
        if not label_dir.exists():
            print(f"No se encontraron etiquetas en: {label_dir}")
            continue

        txt_files = list(label_dir.glob("*.txt"))
        detections = [f for f in txt_files if f.stat().st_size > 0]
        total = len(txt_files)

        # Métricas agregadas
        class_counts = defaultdict(int)
        class_confidences = defaultdict(list)

        for file in detections:
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_id = int(parts[0])
                        conf = float(parts[5])
                        class_counts[cls_id] += 1
                        class_confidences[cls_id].append(conf)

        # Resultados globales
        row = {
            "Modelo": model,
            "Dificultad": category,
            "Total Imágenes": total,
            "Con Detecciones": len(detections),
            "(%) Con detección": round(100 * len(detections) / total, 2) if total > 0 else 0,
            "Media detecciones/img": round(sum(class_counts.values()) / total, 2) if total > 0 else 0
        }

        # Añadir métricas por clase
        for cls_id, name in class_names.items():
            row[f"{name} detecciones"] = class_counts[cls_id]
            row[f"{name} conf media"] = round(
                sum(class_confidences[cls_id]) / len(class_confidences[cls_id]), 3
            ) if class_confidences[cls_id] else 0.0

        summary.append(row)

# DataFrame y CSV
df = pd.DataFrame(summary)
csv_path = base_dir / "comparacion_detecciones_completa.csv"
df.to_csv(csv_path, index=False)

print("\nComparación extendida de Detecciones:")
print(df[["Modelo", "Dificultad", "(%) Con detección", "Media detecciones/img"]].to_string(index=False))

# Calcular puntuación ponderada por clase
for cls_id, name in class_names.items():
    max_det = df[f"{name} detecciones"].max()
    norm_dets = df[f"{name} detecciones"] / max_det if max_det > 0 else 0
    confs = df[f"{name} conf media"]

    # Score = detecciones normalizadas * 0.6 + confianza * 0.4
    df[f"{name} score"] = (norm_dets * 0.6 + confs * 0.4).round(3)

# Calcular un score total por modelo
score_cols = [c for c in df.columns if c.endswith("score")]
df["Score total ponderado"] = df[score_cols].mean(axis=1).round(3)

# Ordenar por mejor score
df = df.sort_values(by="Score total ponderado", ascending=False)

# Guardar también el ranking ordenado por score total
ranking_path = base_dir / "ranking_score_total.csv"
ranking_df = df[["Modelo", "Dificultad", "Score total ponderado"]]
ranking_df.to_csv(ranking_path, index=False)

print(f"\nRanking por Score Total Ponderado guardado en: {ranking_path}")
print("\nRanking por Score Total Ponderado:")
print(df[["Modelo", "Dificultad", "Score total ponderado"]].to_string(index=False))

# Gráficos
plot_dir = base_dir / "graficos"
plot_dir.mkdir(parents=True, exist_ok=True)

# % detección
plt.figure(figsize=(10, 6))
for cat in categories:
    subset = df[df["Dificultad"] == cat]
    plt.bar(subset["Modelo"] + f" ({cat})", subset["(%) Con detección"], label=cat)
plt.title("(%) de imágenes con detecciones")
plt.ylabel("%")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(plot_dir / "porcentaje_detecciones.png")
plt.close()

# Media detecciones por imagen
plt.figure(figsize=(10, 6))
for cat in categories:
    subset = df[df["Dificultad"] == cat]
    plt.bar(subset["Modelo"] + f" ({cat})", subset["Media detecciones/img"], label=cat)
plt.title("Media de detecciones por imagen")
plt.ylabel("Detecciones")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(plot_dir / "media_detecciones.png")
plt.close()

print(f"\nCSV guardado en: {csv_path}")
print(f"Gráficos en: {plot_dir}")