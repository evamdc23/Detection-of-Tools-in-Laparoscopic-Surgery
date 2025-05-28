from pathlib import Path
from typing import List, Tuple, Optional
import logging
import json
import cv2
import numpy as np
from .base_preprocessor import BasePreprocessor
from .dataset_unifier import DatasetUnifier
from config.paths import paths

"""
Este módulo implementa la clase CholecTrack20Preprocessor, una subclase de BasePreprocessor, que se encarga de preprocesar 
el dataset CholecTrack20. Este dataset sí contiene anotaciones completas con bounding boxes, por lo que se generan etiquetas 
en formato YOLO estándar con coordenadas reales para cada herramienta quirúrgica detectada.
"""

class CholecTrack20Preprocessor(BasePreprocessor):
    def __init__(self, weak_supervision: bool = False, sampling_step: int = 5):
        self.sampling_step = sampling_step
        self.weak_supervision = weak_supervision
        self.unifier = DatasetUnifier(enhance_bboxes=weak_supervision)
        super().__init__("cholectrack20_for_val")
        self.label_cache = {}

    def _get_tool_classes(self) -> List[str]:
        return self.unifier.unified_classes  # Usa los nombres unificados de herramientas

    def _load_json_annotations(self, video_id: str) -> Optional[dict]:
        # Busca el JSON en los tres splits posibles
        for split in ["Training", "Validation", "Testing"]:
            candidate = paths.DATASETS["cholectrack20"][split] / video_id / f"{video_id}.json"
            if candidate.exists():
                with open(candidate, "r") as f:
                    data = json.load(f)
                    return data.get("annotations", {})
        logging.warning(f"No se encontró JSON para video {video_id}")
        return None

    def _create_yolo_annotation(self, frame_idx: int, video_path: Path, frame_shape: Tuple[int, int]) -> List[str]:
        video_id = video_path.parents[1].name
        h, w = frame_shape
        if video_id not in self.label_cache:
            self.label_cache[video_id] = self._load_json_annotations(video_id)
        if self.label_cache[video_id] is None:
            return []
        annotations = self.label_cache.get(video_id, {}).get(str(frame_idx), [])
        yolo_lines = []

        for ann in annotations:
            class_id = ann.get("instrument", -1)
            bbox = ann.get("tool_bbox", [])
            if class_id == -1 or not bbox:
                continue
            x, y, bw, bh = bbox
            cx = x + bw / 2
            cy = y + bh / 2
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return yolo_lines

    def get_valid_frame_indices(self, total_frames: int, fps: float, video_path: Path) -> List[int]:
        video_id = video_path.name
        annotations = self._load_json_annotations(video_id)
        if not annotations:
            return []

        """
        sampling_step = 1: usas todos los frames anotados (100%)
        sampling_step = 5: usas 1 de cada 5 frames anotados (20%)
        sampling_step = 10: usas 1 de cada 10 (10%)
        """

        annotated_frames = sorted(map(int, annotations.keys()))
        sampled = annotated_frames[::self.sampling_step]
        return sampled
    
    def process_video(self, video_path: Path) -> int:
        video_id = video_path.name
        frame_dir = video_path / "Frames"
        if not frame_dir.exists():
            logging.warning(f"No se encontró carpeta 'Frames' para {video_id}")
            return 0

        all_frames = sorted(frame_dir.glob("*.png"))
        processed = 0
        for img_path in all_frames:
            frame_idx = int(img_path.stem)
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            out_img_path = self._save_frame(frame, video_id, frame_idx)
            annotations = self._create_yolo_annotation(frame_idx, img_path, frame.shape[:2])
            if annotations:
                self._save_yolo_label(out_img_path, annotations)
                processed += 1

        logging.info(f"{processed} frames procesados para {video_id}")
        return processed
