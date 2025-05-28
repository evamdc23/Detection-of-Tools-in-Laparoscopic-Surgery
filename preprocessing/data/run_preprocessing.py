import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Añadir raíz del proyecto
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from config.paths import paths
from data.preprocessing.cholec80 import Cholec80Preprocessor
from data.preprocessing.m2cai16 import M2CAI16Preprocessor
from data.preprocessing.surgtoolloc import SurgToolLocPreprocessor
from data.preprocessing.cholectrack20 import CholecTrack20Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Este script actúa como punto de entrada para ejecutar el preprocesamiento de diferentes datasets quirúrgicos. 
Se encarga de seleccionar el preprocesador adecuado según el nombre del dataset, cargar los videos correspondientes 
y procesarlos uno por uno, generando frames y anotaciones YOLO en función de la configuración.
"""

def get_preprocessor(dataset_name: str, version: str = None, weak: bool = False, sampling_step: int = 5):
    if dataset_name == "cholec80":
        return Cholec80Preprocessor(weak_supervision=weak), [paths.DATASETS["cholec80"]["videos"]]
    elif dataset_name == "m2cai16":
        return M2CAI16Preprocessor(weak_supervision=weak), [
            paths.DATASETS["m2cai16"]["videos"]["train"],
            paths.DATASETS["m2cai16"]["videos"]["test"]
        ]
    elif dataset_name == "surgtoolloc":
        return SurgToolLocPreprocessor(weak_supervision=weak), [paths.DATASETS["surgtoolloc"]["videos"]]
    elif dataset_name == "cholectrack20":
        return CholecTrack20Preprocessor(weak_supervision=weak, sampling_step=sampling_step), [
            paths.DATASETS["cholectrack20"]["Training"],
            paths.DATASETS["cholectrack20"]["Validation"],
            paths.DATASETS["cholectrack20"]["Testing"]
        ]
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")

def run_dataset_preprocessing(dataset_name: str, version: str = None, max_videos: int = None, weak: bool = False, subset_csv: Path = None, sampling_step: int = 5):
    processor, video_dirs = get_preprocessor(dataset_name, version, weak)
    all_videos = []

    if dataset_name == "surgtoolloc":
        if subset_csv:
            df_subset = pd.read_csv(subset_csv)
            selected_names = set(df_subset["clip_name"])
            video_dir = video_dirs[0]
            all_videos = [video_dir / f"{name}.mp4" for name in selected_names if (video_dir / f"{name}.mp4").exists()]
        else:
            for video_dir in video_dirs:
                all_videos.extend(sorted(video_dir.glob("clip_*.mp4")))
    elif dataset_name == "cholectrack20":
        for split_dir in video_dirs:
            for video_folder in sorted(split_dir.glob("VID*")):
                if (video_folder / "Frames").exists():
                    all_videos.append(video_folder)
    else:
        for video_dir in video_dirs:
            all_videos.extend(sorted(video_dir.glob("*.mp4")))

    if not all_videos:
        logging.error(f"No se encontraron videos para el dataset {dataset_name}")
        return

    if max_videos:
        all_videos = all_videos[:max_videos]

    logging.info(f"Procesando {len(all_videos)} videos del dataset '{dataset_name}'")

    for i, video in enumerate(all_videos):
        logging.info(f"[{i+1}] Procesando: {video.name} ({video})")
        try:
            n_frames = processor.process_video(video)
            logging.info(f" -> {n_frames} frames procesados.")
        except Exception as e:
            logging.error(f"Error procesando {video.name}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset")
    parser.add_argument("--max-videos", type=int, help="Cantidad máxima de videos a procesar")
    parser.add_argument("--weak-supervision", action="store_true", help="Activar supervisión débil para pseudo-bboxes")
    parser.add_argument("--subset-csv", type=Path, help="Ruta a CSV con lista de clip_name (solo para surgtoolloc)")
    parser.add_argument("--sampling-step", type=int, default=5, help="Extraer 1 de cada N frames anotados (solo para cholectrack20)")
    args = parser.parse_args()

    run_dataset_preprocessing(args.dataset, args.version, args.max_videos, args.weak_supervision, args.subset_csv, args.sampling_step)