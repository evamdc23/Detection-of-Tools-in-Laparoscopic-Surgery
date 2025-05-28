import os
from pathlib import Path
import cv2
import albumentations as A
from collections import Counter

"""
Este script realiza data augmentation específicamente sobre imágenes que contienen herramientas 
quirúrgicas clasificadas como clases minoritarias, con el fin de balancear el conjunto de entrenamiento para YOLO.
"""

# Clases minoritarias
RARE_CLASSES = {"irrigator", "clipper", "specimen_bag", "bipolar", "scissors"}
CLASS_NAME_TO_ID = {
    "grasper": 0,
    "bipolar": 1,
    "hook": 2,
    "scissors": 3,
    "clipper": 4,
    "irrigator": 5,
    "specimen_bag": 6,
}
ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# Configuración de paths
IMG_DIR = Path("processed/cholectrack20_split/train/images")
LABEL_DIR = Path("processed/cholectrack20_split/train/labels")

# Transformaciones
transform = A.Compose([
    A.Rotate(limit=30, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], check_each_transform=False))

# Cargar etiquetas en formato YOLO
def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

# Validar y filtrar bboxes válidos antes de transformar
def prefilter_valid_bboxes(labels):
    result = []
    for l in labels:
        try:
            cls = int(float(l[0]))
            x, y, w, h = map(float, l[1:])
            if 0 <= x <= 1 and 0 <= y <= 1 and w > 0.01 and h > 0.01 \
                and x - w/2 >= 0 and x + w/2 <= 1 and y - h/2 >= 0 and y + h/2 <= 1:
                result.append((cls, [x, y, w, h]))
        except Exception:
            continue
    return result

# Buscar si alguna clase es minoritaria en los labels
def has_rare_class(labels):
    try:
        return any(int(float(l[0])) in [CLASS_NAME_TO_ID[c] for c in RARE_CLASSES] for l in labels)
    except Exception:
        return False

augmented_counter = Counter()
image_counter = 0

for label_path in LABEL_DIR.glob("*.txt"):
    labels = load_labels(label_path)
    if not has_rare_class(labels):
        continue

    valid = prefilter_valid_bboxes(labels)
    if not valid:
        continue

    img_path = IMG_DIR / (label_path.stem + ".jpg")
    if not img_path.exists():
        continue

    image = cv2.imread(str(img_path))
    class_ids = [cls for cls, _ in valid]
    bboxes = [box for _, box in valid]

    for i in range(3):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        except Exception as e:
            print(f"[SKIPPED] {label_path.stem} (i={i}) → {e}")
            continue

        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        filtered = [(cls, box) for cls, box in zip(aug_labels, aug_bboxes)
                    if 0 <= box[0] <= 1 and 0 <= box[1] <= 1 and 0 < box[2] <= 1 and 0 < box[3] <= 1
                    and box[0] - box[2]/2 >= 0 and box[0] + box[2]/2 <= 1
                    and box[1] - box[3]/2 >= 0 and box[1] + box[3]/2 <= 1]

        if not filtered:
            continue

        new_name = f"aug_{label_path.stem}_{i}"
        aug_img_path = IMG_DIR / f"{new_name}.jpg"
        aug_lbl_path = LABEL_DIR / f"{new_name}.txt"

        cv2.imwrite(str(aug_img_path), aug_image)
        with open(aug_lbl_path, 'w') as f:
            for cls, box in filtered:
                f.write(f"{cls} {' '.join([str(round(b, 6)) for b in box])}\n")
                augmented_counter[cls] += 1

        image_counter += 1

# Resumen final
print("\nAumentación completada.")
print(f"\nNuevas imágenes creadas: {image_counter}")
print("\nObjetos aumentados por clase:")
for cls_id, count in augmented_counter.items():
    print(f"- {ID_TO_CLASS_NAME[cls_id]}: {count} bboxes")
