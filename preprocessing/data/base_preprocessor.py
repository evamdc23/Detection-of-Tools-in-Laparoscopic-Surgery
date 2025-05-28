from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import random
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)

"""
Este script define una clase abstracta llamada BasePreprocessor, que sirve como plantilla 
base para preprocesadores específicos de distintos datasets de video quirúrgico.

Está diseñada para ser extendida por subclases, que deben implementar métodos personalizados 
para extraer clases de herramientas quirúrgicas, procesar videos y generar anotaciones en 
formato YOLO.
"""

class BasePreprocessor(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.output_dir = Path(f"processed/{dataset_name}")
        self.tool_classes = self._get_tool_classes()
    
    @abstractmethod
    def _get_tool_classes(self) -> List[str]:
        # Obtener lista de clases de herramientas para este dataset
        pass

    @abstractmethod
    def process_video(self, video_path: Path) -> int:
        # Procesar un solo video
        pass

    @abstractmethod
    def _create_yolo_annotation(self, frame_idx: int, video_path: Path, frame_shape: Tuple[int, int]) -> List[str]:
        # Retorna una lista de strings en formato YOLO [clase x_center y_center widith height]
        # o una lista vacía si no hay anotaciones
        pass

    def get_valid_frame_indices(self, total_frames: int, fps: float, video_path: Path) -> List[int]:
        """
        Puede ser sobreescrito por subclases para restringir los índices válidos a los que tienen anotaciones.
        """
        return self._get_frame_indices(total_frames / fps, fps)


    def extract_frames(self, video_path: Path) -> List[Tuple[Path, int]]:
        """ Función compartida para extraer frames:
        - Videos <1 min: 5 frames equidistantes
        - Videos 1-10 min: 2 frames cada 30 seg
        - Videos >10 min: Cortar a 10 min y aplicar regla anterior 
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0:
            raise ValueError(f"FPS inválido en el video: {video_path}")

        frame_indices = self.get_valid_frame_indices(total_frames, fps, video_path)
        saved_frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue

            annotation = self._create_yolo_annotation(idx, video_path, frame.shape[:2])
            if annotation:
                frame_path = self._save_frame(frame, video_path.stem, idx)
                self._save_yolo_label(frame_path, annotation)
                saved_frames.append((frame_path, idx))  # Guardar solo si hay label

        cap.release()
        return saved_frames
    
    def _get_frame_indices(self, duration: float, fps: float) -> List[int]:
        if duration < 60:
            return sorted(random.sample(range(int(duration * fps)), min(5, int(duration * fps))))
        else:
            max_duration = min(duration, 600)
            indices = []
            for sec in range(0, int(max_duration), 30):
                indices.append(int(sec * fps))
                indices.append(int((sec + 15) * fps))
            return indices
    
    def _save_frame(self, frame: np.ndarray, video_stem: str, frame_idx: int) -> Path:
        # Extrae video_id tipo 'video01'
        video_id = video_stem.lower()
        save_dir = self.output_dir / "frames" / video_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Guardar frame
        frame_path = save_dir / f"{video_stem}_frame_{frame_idx:05d}.jpg"         
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    
    def _save_yolo_label(self, frame_path: Path, yolo_lines: List[str]):
        # Extraer nombre de subcarpeta (video_id) desde frame_path
        video_id = frame_path.parent.name
        label_dir = self.output_dir / "labels" /video_id
        label_dir.mkdir(parents=True, exist_ok=True)

        # Guardar archivo de anotaciones
        label_path = label_dir / f"{frame_path.stem}.txt"
        with open(label_path, "w") as f:
            f.writelines([line + "\n" for line in yolo_lines])