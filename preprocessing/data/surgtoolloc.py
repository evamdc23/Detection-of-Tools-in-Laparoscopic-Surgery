from pathlib import Path
from typing import List, Dict
import logging
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_preprocessor import BasePreprocessor
from .dataset_unifier import DatasetUnifier
from config.paths import paths

"""
La clase SurgToolLocPreprocessor es una subclase de BasePreprocessor diseñada para procesar el dataset SurgToolLoc, 
que está compuesto por clips de video de 30 segundos anotados con la presencia global de herramientas quirúrgicas 
(no por frame, ni con coordenadas). El preprocesador genera etiquetas en formato YOLO para cada frame extraído, 
con supervisión débil opcional.
"""

class SurgToolLocPreprocessor(BasePreprocessor):
    def __init__(self, enable_parallel: bool = True, max_workers: int = 4, weak_supervision: bool = False):
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.weak_supervision = weak_supervision

        self.unifier = DatasetUnifier(enhance_bboxes=self.weak_supervision)

        super().__init__("surgtoolloc")

        self.clip_duration = 30  # segundos por clip
        self.fps = 30  # FPS del video original
        self.frame_interval = 30  # Extraer 1 frame por segundo

        self.labels_df = self._load_labels()

    def _load_labels(self) -> pd.DataFrame:
        labels_path = paths.DATASETS['surgtoolloc']['labels']
        df = pd.read_csv(labels_path)

        def parse_tools(tools_str):
            try:
                tools_str = tools_str.strip("[]")
                tools = [t.strip().strip("'\"") for t in tools_str.split(",")]
                return [t for t in tools if t.lower() != "nan"]
            except Exception as e:
                logging.warning(f"Error parsing tools: {tools_str} | {e}")
                return []

        df["tools_present"] = df["tools_present"].apply(parse_tools)
        return df.set_index("clip_name")

    def _get_tool_classes(self) -> List[str]:
        return self.unifier.unified_classes

    def _create_yolo_annotation(self, frame: np.ndarray, tools: List[str]) -> List[str]:
        annotations = []
        for tool in tools:
            unified_tool = self.unifier.unify_tool_name(tool)
            class_id = self.unifier.get_unified_class_id(unified_tool)
            if class_id is not None:
                if self.weak_supervision:
                    bbox = self.unifier.enhance_weak_supervision(frame, unified_tool)
                else:
                    bbox = [0.5, 0.5, 1.0, 1.0]
                annotations.append(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}")
        return annotations

    def process_video(self, video_path: Path):
        video_id = video_path.stem
        logging.info(f"Procesando video: {video_id}")

        frames_out_path = paths.get_processed_frames_path("surgtoolloc", video_id)
        labels_out_path = paths.PROCESSED_SUBDIRS["surgtoolloc"] / "labels" / video_id
        labels_out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning(f"No se pudo abrir el video {video_path}")
            return 0

        frame_count = 0
        saved_frames = 0
        tools = self.labels_df.loc[video_id]["tools_present"]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                frame_name = f"{frame_count:06d}.jpg"
                frame_path = frames_out_path / frame_name
                cv2.imwrite(str(frame_path), frame)

                # Guardar etiqueta YOLO
                annotation_txt = self._create_yolo_annotation(frame, tools)
                label_file = labels_out_path / frame_name.replace(".jpg", ".txt")
                with open(label_file, "w") as f:
                    f.write("\n".join(annotation_txt))

                saved_frames += 1

            frame_count += 1

        cap.release()

        logging.info(f"Frames guardados en: {frames_out_path}")
        logging.info(f"Labels guardadas en: {labels_out_path}")
        logging.info(f" -> {saved_frames} frames procesados")
        return saved_frames

    def preprocess(self):
        videos_dir = paths.DATASETS["surgtoolloc"]["videos"]
        all_videos = list(videos_dir.glob("*.mp4"))

        if self.enable_parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_video, video) for video in all_videos]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando videos"):
                    future.result()
        else:
            for video in tqdm(all_videos, desc="Procesando videos"):
                self.process_video(video)