import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor
from .dataset_unifier import DatasetUnifier
from config.paths import paths

"""
Este módulo implementa la clase Cholec80Preprocessor, una subclase de BasePreprocessor, que se encarga 
de preprocesar los videos del dataset Cholec80 generando anotaciones en formato YOLO a partir de 
etiquetas binarias por frame (etiquetas de presencia/ausencia de herramientas quirúrgicas). 

Este enfoque se basa en supervisión débil, ya que no hay bounding boxes.
"""

class Cholec80Preprocessor(BasePreprocessor):
    # Definimos el orden y nombres de las herramientas, este orden define el class_id en YOLO
    required_tools = ['Grasper', 'Bipolar', 'Hook', 'Scissors',
                      'Clipper', 'Irrigator', 'SpecimenBag']

    def __init__(self, weak_supervision: bool = False):
        # Inicializa el preprocesador con el nombre del dataset
        super().__init__("cholec80")
        self.unifier = DatasetUnifier(enhance_bboxes=weak_supervision)
        # Ruta a la carpeta donde están los .txt de anotaciones de herramientas
        self.tool_annotations_dir = paths.DATASETS["cholec80"]["tool_annotations"]
        # Caché para no volver a cargar el mismo archivo varias veces si se reutiliza
        self.labels_cache: dict = {}

    def _get_tool_classes(self) -> List[str]:
        # Retorna la lista de clases de herramientas en orden fijo
        return self.required_tools

    def _extract_video_id(self, video_path: Path) -> Optional[str]:
        # Extrae el número del video desde su nombre, por ejemplo: "video01.mp4" -> "01"
        name = video_path.stem.lower()
        if "video" in name:
            return name.replace("video", "").zfill(2)
        return None

    def _load_tool_annotations(self, video_id: str) -> Optional[pd.DataFrame]:
        # Construye la ruta al archivo de anotaciones de herramientas
        annotation_file = self.tool_annotations_dir / f"video{video_id}-tool.txt"
        if not annotation_file.exists():
            logging.warning(f"No se encontró el archivo de anotaciones: {annotation_file}")
            return None
        try:
            # Usa delim_whitespace para manejar encabezados con espacios irregulares
            df = pd.read_csv(annotation_file, sep=r'\s+')
            # Se asegura que la columna 'Frame' sea int por si acaso
            df["Frame"] = df["Frame"].astype(int)
            return df
        except Exception as e:
            logging.error(f"Error al leer {annotation_file}: {e}")
            return None

    def _create_yolo_annotation(self, frame_idx: int, video_path: Path, frame_shape: Tuple[int, int]) -> List[str]:
        """
        Genera anotaciones YOLO para un frame dado:
        Si hay herramientas presentes, retorna una línea YOLO por herramienta, centrada en la imagen.
        En este dataset no hay bounding boxes, así que usamos etiquetas de presencia (weak supervision).
        """
        # Extrae el ID del video, por ejemplo '01'
        video_id = self._extract_video_id(video_path)
        if not video_id:
            return []

        # Si aún no cargamos el archivo de anotaciones, lo hacemos y lo almacenamos en caché
        if video_id not in self.labels_cache:
            df = self._load_tool_annotations(video_id)
            if df is None:
                return []
            self.labels_cache[video_id] = df

        # Recuperamos el DataFrame con las anotaciones
        df = self.labels_cache[video_id]

        # Buscamos la fila que corresponde exactamente al frame actual
        row = df[df["Frame"] == frame_idx]
        if row.empty:
            return []

        # Preparamos la lista de anotaciones YOLO que devolveremos
        annotations = []

        # Recorremos las herramientas en el orden definido
        for class_id, tool in enumerate(self.tool_classes):
            # Si está presente (== 1), creamos una anotación "centrada" en el frame (formato YOLO)
            if row.iloc[0][tool] == 1:
                annotations.append(f"{class_id} 0.5 0.5 1.0 1.0")

        return annotations

    def get_valid_frame_indices(self, total_frames: int, fps: float, video_path: Path) -> List[int]:
        video_id = self._extract_video_id(video_path)
        annotated_frames = self.get_annotated_frame_indices(video_id)
        if not annotated_frames:
            return []

        # Aplicar el sampling SOLO sobre frames con anotación
        duration = total_frames / fps
        candidate_indices = self._get_frame_indices(duration, fps)
        valid = [idx for idx in candidate_indices if int(idx) in annotated_frames]
        return valid

    def get_annotated_frame_indices(self, video_id: str) -> List[int]:
        df = self._load_tool_annotations(video_id)
        if df is None:
            return []
        return df["Frame"].astype(int).tolist()


    def process_video(self, video_path: Path) -> int:
        """
        Procesa un video: extrae frames y genera sus respectivas etiquetas en formato YOLO.
        Retorna el número de frames extraídos y procesados.
        """
        try:
            # Este método está definido en la clase base: extrae frames y guarda anotaciones
            frames = self.extract_frames(video_path)
            logging.info(f"Procesado {len(frames)} frames del video {video_path.name}")
            return len(frames)
        except Exception as e:
            logging.error(f"Error procesando {video_path.name}: {e}")
            return 0