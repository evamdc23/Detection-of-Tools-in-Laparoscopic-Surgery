from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import logging
import re
from .base_preprocessor import BasePreprocessor
from .dataset_unifier import DatasetUnifier
from config.paths import paths
import cv2

"""
El módulo implementa la clase M2CAI16Preprocessor, una subclase de BasePreprocessor, 
diseñada para procesar los videos del dataset M2CAI16. Este preprocesador genera 
anotaciones en formato YOLO basadas en la presencia binaria de herramientas quirúrgicas por frame. 
Como no se proporcionan bounding boxes reales, se utiliza un enfoque de supervisión débil, 
creando cajas sintéticas centradas en el frame.
"""

class M2CAI16Preprocessor(BasePreprocessor):
    required_tools = ['Grasper', 'Bipolar', 'Hook', 'Scissors',
                      'Clipper', 'Irrigator', 'SpecimenBag']

    def __init__(self, weak_supervision: bool = False):
        super().__init__("m2cai16")
        self.weak_supervision = weak_supervision
        self.unifier = DatasetUnifier(enhance_bboxes=weak_supervision)
        self.labels_cache = {}
        self.annotations = {
            "train": paths.DATASETS["m2cai16"]["annotations"]["train"],
            "test": paths.DATASETS["m2cai16"]["annotations"]["test"]
        }

    def _get_tool_classes(self) -> List[str]:
        return self.required_tools

    def _extract_video_id(self, video_path: Path) -> Optional[str]:
        match = re.search(r'tool_video_(\d+)', video_path.stem)
        return match.group(1) if match else None

    def _load_tool_annotations(self, video_id: str) -> Optional[pd.DataFrame]:
        vid_num = int(video_id)
        split = "train" if vid_num <= 10 else "test"
        annotation_file = self.annotations[split] / f"tool_video_{video_id}.txt"

        if not annotation_file.exists():
            logging.warning(f"No se encontró el archivo de anotaciones: {annotation_file}")
            return None

        try:
            df = pd.read_csv(annotation_file, sep=r'\s+')
            df["Frame"] = df["Frame"].astype(int)
            return df
        except Exception as e:
            logging.error(f"Error leyendo anotaciones para video {video_id}: {e}")
            return None

    def _create_yolo_annotation(self, frame_idx: int, video_path: Path, frame_shape: Tuple[int, int]) -> List[str]:
        video_id = self._extract_video_id(video_path)
        if not video_id:
            return []

        if video_id not in self.labels_cache:
            df = self._load_tool_annotations(video_id)
            if df is None:
                return []
            self.labels_cache[video_id] = df

        df = self.labels_cache[video_id]
        row = df[df["Frame"] == frame_idx]
        if row.empty:
            return []

        annotations = []
        for class_id, tool in enumerate(self.tool_classes):
            if tool in row.columns and row.iloc[0][tool] == 1:
                annotations.append(f"{class_id} 0.5 0.5 1.0 1.0")

        return annotations

    def _save_frame(self, frame: cv2.typing.MatLike, video_stem: str, frame_idx: int) -> Path:
        # Fuerza video_id como 'video01' desde 'tool_video_01'
        video_id = self._extract_video_id(Path(video_stem))
        folder_name = f"video{video_id}"
        save_dir = self.output_dir / "frames" / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_path = save_dir / f"{video_stem}_frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        return frame_path

    def _save_yolo_label(self, frame_path: Path, yolo_lines: List[str]):
        video_id = self._extract_video_id(frame_path)
        label_dir = self.output_dir / "labels" / f"video{video_id}"
        label_dir.mkdir(parents=True, exist_ok=True)

        label_path = label_dir / f"{frame_path.stem}.txt"
        with open(label_path, "w") as f:
            f.writelines([line + "\n" for line in yolo_lines])

    def get_valid_frame_indices(self, total_frames: int, fps: float, video_path: Path) -> List[int]:
        # Extraer ID del video, ej: tool_video_01 -> 01
        video_name = video_path.stem
        video_id = video_name.replace("tool_video_", "").zfill(2)
        split = "train" if "train" in str(video_path) else "test"

        annotated = self.get_annotated_frame_indices(f"tool_video_{video_id}", split)
        if not annotated:
            return []

        duration = total_frames / fps
        candidate_indices = self._get_frame_indices(duration, fps)
        valid = [idx for idx in candidate_indices if int(idx) in annotated]
        return valid


    def get_annotated_frame_indices(self, video_id: str, split: str) -> List[int]:
        annotation_path = self.annotations[split] / f"{video_id}.txt"
        if not annotation_path.exists():
            return []
        try:
            df = pd.read_csv(annotation_path, sep=r'\s+')
            return df["Frame"].astype(int).tolist()
        except Exception as e:
            logging.error(f"Error al leer anotaciones de {annotation_path}: {e}")
            return []

    def process_video(self, video_path: Path) -> int:
        try:
            frames = self.extract_frames(video_path)
            video_id = self._extract_video_id(video_path)
            output_dir = self.output_dir.resolve()
            logging.info(f"Procesado {len(frames)} frames del video {video_path.name}")
            logging.info(f"Frames guardados en: {output_dir}/frames/video{video_id}")
            logging.info(f"Labels guardadas en: {output_dir}/labels/video{video_id}")
            return len(frames)
        except Exception as e:
            logging.error(f"Error procesando {video_path.name}: {e}")
            return 0