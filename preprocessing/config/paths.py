from pathlib import Path
from typing import Literal, Dict, Any
import os

"""
Este script define una clase Paths que centraliza, gestiona y valida las rutas necesarias para 
el procesamiento de videos quirúrgicos. Está diseñado para facilitar el acceso y control de directorios de datasets, 
salidas procesadas, modelos entrenados y resultados de experimentos.
"""

# Nombres de datasets
DatasetName = Literal['cholec80', 'cholect50', 'cholect50_challenge', 'm2cai16', 'surgtoolloc']

class Paths:
    def __init__(self):
        # 1. Directorio base del proyecto
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        # 2. Rutas de datasets
        self._init_dataset_paths()

        # 3. Rutas de output
        self._init_output_paths()

        # Validación y creación de directorios
        self.create_dirs()

    def _init_dataset_paths(self):
        self.DATASETS = {
            'cholec80': {
                'videos': self.BASE_DIR / "../videos_cholec80/videos", 
                'tool_annotations': self.BASE_DIR / "../videos_cholec80/tool_annotations",
                'phase_annotations': self.BASE_DIR / "../videos_cholec80/phase_annotations"
            },
            # Versión 2023 (Full - 50 videos)
            'cholect50': {
                'videos': self.BASE_DIR / "../videos_cholect50/videos",
                'labels': self.BASE_DIR / "../videos_cholect50/labels",
                'mapping': self.BASE_DIR / "../videos_cholect50/label_mapping.txt"
            },
            # Versión 2022 (Challenge Validation - 5 videos)
            'cholect50_challenge': {
                'videos': self.BASE_DIR / "../videos_cholect50_challenge_val/videos",
                'labels': self.BASE_DIR / "../videos_cholect50_challenge_val/labels",
                'mapping': self.BASE_DIR / "../videos_cholect50_challenge_val/label_mapping.txt"
            },
            'm2cai16': {
                'videos': {
                    'train': self.BASE_DIR / "../videos_m2cai16_tool_locations/train_dataset",
                    'test': self.BASE_DIR / "../videos_m2cai16_tool_locations/test_dataset"
                },
                'annotations' : {
                    'train': self.BASE_DIR / "../videos_m2cai16_tool_locations/train_dataset",
                    'test': self.BASE_DIR / "../videos_m2cai16_tool_locations/test_dataset"
                }
            },
            'surgtoolloc': {
                'videos': self.BASE_DIR / "../videos_surgtoolloc",
                'labels': self.BASE_DIR / "../videos_surgtoolloc/labels.csv"
            },
            'cholectrack20': {
                'Training': self.BASE_DIR / "../videos_cholectrack20/Training",
                'Validation': self.BASE_DIR / "../videos_cholectrack20/Validation",
                'Testing': self.BASE_DIR / "../videos_cholectrack20/Testing"
            }
        }

        # Resuelve las rutas para obtener paths absolutos
        for dataset in self.DATASETS.values():
            for key, path in list(dataset.items()):
                if isinstance(path, Path):
                    dataset[key] = path.resolve()
                elif isinstance(path, Dict):
                    # Manejar subdiccionarios
                    for subkey, subpath in path.items():
                        if isinstance(subpath, Path):
                            path[subkey] = subpath.resolve()

    def _init_output_paths(self):
        self.PROCESSED_DIR = self.BASE_DIR / "processed"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.RESULTS_DIR = self.BASE_DIR / "results"

        # Subdirectorios para datasets procesados (se crearán automáticamente)
        self.PROCESSED_SUBDIRS = {
            'cholec80': self.PROCESSED_DIR / "cholec80",
            'cholect50': self.PROCESSED_DIR / "cholect50",
            'cholect50_challenge': self.PROCESSED_DIR / "cholect50_challenge",
            'm2cai16': self.PROCESSED_DIR / "m2cai16",
            'surgtoolloc': self.PROCESSED_DIR / "surgtoolloc",
            'cholectrack20': self.PROCESSED_DIR / "cholectrack20",
            'mixed': self.PROCESSED_DIR / "mixed",
            'reduced_mixed': self.PROCESSED_DIR / "reduced_mixed"
        }

        # Subrutas dentro de cada dataset preprocesado
        self.PROCESSED_FRAMES = {
            ds: self.PROCESSED_SUBDIRS[ds] / "frames" for ds in self.PROCESSED_SUBDIRS
        }
        self.PROCESSED_LABELS = {
            ds: self.PROCESSED_SUBDIRS[ds] / "labels" for ds in self.PROCESSED_SUBDIRS
        }
        
    def create_dirs(self):
        # Crea las carpetas necesarias para outputs
        try:
            # Directorios principales
            self.PROCESSED_DIR.mkdir(exist_ok=True)
            self.MODELS_DIR.mkdir(exist_ok=True)
            self.RESULTS_DIR.mkdir(exist_ok=True)

            # Subdirectorios para cada dataset
            for path in self.PROCESSED_SUBDIRS.values():
                path.mkdir(exist_ok=True, parents=True)
        
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")
    
    def get_processed_frames_path(self, dataset_name: DatasetName, video_id: str, create_dir: bool = True) -> Path:        
        frames_dir = self.PROCESSED_SUBDIRS[dataset_name] / "frames" / video_id
        if create_dir:
            frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def validate_paths(self, verbose: bool = True) -> bool:
        # Verifica que las rutas de los datasets existen y sean accesibles
        all_valid = True

        # Verificar datasets de entrada
        for ds_name, paths in self.DATASETS.items():
            for path_type, path in paths.items():
                if isinstance(path, dict): # Manejar estructuras anidadas como en m2cai16
                    for sub_path in path.values():
                        if not sub_path.exists():
                            if verbose:
                                print(f"[{ds_name}] Missing: {path_type} at {sub_path}")
                            all_valid = False
                        elif not os.access(sub_path, os.R_OK):
                            if verbose:
                                print(f"[{ds_name}] No read permissions: {sub_path}")
                            all_valid = False
                else:
                    if not path.exists():
                        if verbose:
                           print(f"[{ds_name}] Missing: {path_type} at {path}")
                        all_valid = False
                    elif not os.access(path, os.R_OK):
                        if verbose:
                            print(f"[{ds_name}] No read permissions: {path}")
                        all_valid = False
        
        return all_valid
    
    def check_dataset_structure(self, dataset_name):
        # Verifica que la estructura del dataset sea correcta
        required = {
            'cholec80': ['videos', 'tool_annotations', 'phase_annotations'],
            'cholect50': ['videos', 'labels', 'mapping'],
            'cholect50_challenge': ['videos', 'labels', 'mapping'],
            'm2cai16': ['videos', 'annotations'],
            'surgtoolloc': ['videos', 'labels']
        }

        missing = []
        for key in required.get(dataset_name, []):
            if key not in self.DATASETS[dataset_name]:
                missing.append(key)
                continue
            
            # Manejo especial para m2cai16 con estructura anidada
            if dataset_name == 'm2cai16' and isinstance(self.DATASETS[dataset_name][key], dict):
                for subkey in ['train', 'test']:
                    if subkey not in self.DATASETS[dataset_name][key]:
                        missing.append(f"{key}.{subkey}")
                    elif not self.DATASETS[dataset_name][key][subkey].exists():
                        missing.append(str(self.DATASETS[dataset_name][key][subkey]))
            else:
                if not self.DATASETS[dataset_name][key].exists():
                    missing.append(str(self.DATASETS[dataset_name][key]))

        if missing:
            raise FileNotFoundError(
                f"Estructura incorrecta para {dataset_name}. Faltan: {', '.join(missing)}"
            )
        return True
    
    def get_model_path(self, model_name: str) -> Path:
        # Genera ruta para guardar/leer modelos
        return self.MODELS_DIR / f"{model_name}.pt"
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        # Crea directorio para resultados de experimentos
        exp_dir = self.RESULTS_DIR / experiment_name
        exp_dir.mkdir(exist_ok=True, parents=True)
        return exp_dir

# Instancia global accesible
paths : Paths = None
if not globals().get('paths'):
    paths = Paths()
    if not paths.validate_paths():
        print("Advertencia: Algunas rutas de dataset no son accesibles.")