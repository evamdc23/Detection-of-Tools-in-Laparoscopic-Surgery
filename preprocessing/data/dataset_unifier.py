import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import yaml
from collections import defaultdict

"""
El módulo define la clase DatasetUnifier, encargada de mapear etiquetas de herramientas 
quirúrgicas de distintos datasets a un sistema de clases unificadas, y de generar bounding 
boxes sintéticas (supervisión débil) para entrenamiento con anotaciones incompletas o ausentes.
"""

class DatasetUnifier:
    def __init__(self, enhance_bboxes: bool = True):
        self.enhance_bboxes = enhance_bboxes
        self._init_tool_mappings()
        self._init_bbox_enhancement_params()

    def _init_tool_mappings(self):
        # Mapeo completo de todas las herramientas en todos los datasets
        self.tool_mapping = {
            # Cholec80 y M2CAI16 labels (7 herramientas)
            'Grasper': 'grasper',
            'Bipolar': 'bipolar',
            'Hook': 'hook',
            'Scissors': 'scissors',
            'Clipper': 'clipper',
            'Irrigator': 'irrigator',
            'SpecimenBag': 'specimen_bag',            

            # SurgToolLoc labels (14 herramientas)
            'bipolar dissector': 'bipolar', # Mapea a bipolar
            'bipolar forceps': 'bipolar', # Mapea a bipolar
            'cadiere forceps': 'grasper',
            'clip applier': 'clipper', # Mapea a clipper
            'force bipolar': 'bipolar', # Mapea a bipolar
            'grasping retractor': 'grasper', # Mapea a grasper
            'monopolar curved scissors': 'scissors',
            'needle driver': 'grasper',
            'permanent cautery hook/spatula': 'hook', # Mapea a hook
            'prograsp forceps': 'grasper',
            'stapler': 'other',
            'suction irrigator': 'irrigator', # Mapea a irrigator
            'tip-up fenestrated grasper': 'grasper', # Mapea a grasper
            'vessel sealer': 'bipolar' # Mapea a bipolar
        }

        # Formas alternativas de escritura
        self.alternative_names = {
            'bipolarforceps': 'bipolar forceps',
            'bipolar_forceps': 'bipolar forceps',
            'bipolardissector': 'bipolar dissector',
            'bipolar_dissector': 'bipolar dissector',
            'cadiereforceps': 'cadiere forceps',
            'cadiere_forceps': 'cadiere forceps',
            'clipapplier': 'clip applier',
            'clip_applier': 'clip applier',
            'forcebipolar': 'force bipolar', 
            'force_bipolar': 'force bipolar',
            'graspingretractor': 'grasping retractor',
            'grasping_retractor': 'grasping retractor',
            'monopolarcurvedscissors': 'monopolar curved scissors',
            'monopolar_curved_scissors': 'monopolar curved scissors',
            'needledriver': 'needle driver',
            'needle_driver': 'needle driver',
            'permanentcauteryhook/spatula': 'permanent cautery hook/spatula',
            'permanent_cautery_hook/spatula': 'permanent cautery hook/spatula',
            'prograspforceps': 'prograsp forceps',
            'prograsp_forceps': 'prograsp forceps',
            'suctionirrigator': 'suction irrigator',
            'suction_irrigator': 'suction irrigator',
            'tip-upfenestratedgrasper': 'tip-up fenestrated grasper',
            'tip-up_fenestrated_grasper': 'tip-up fenestrated grasper',
            'vesselsealer': 'vessel sealer',
            'vessel_sealer': 'vessel sealer'
        }

        # Clases finales unificadas (el orden determina el ID de clase)
        self.unified_classes = [
            'grasper', # 0
            'bipolar', # 1
            'hook', # 2
            'scissors', # 3
            'clipper',  # 4 
            'irrigator', # 5
            'specimen_bag', # 6
            'other' # 7
        ]

        # Parámetros para mejorar bounding boxes por tipo de herramienta
        self.bbox_params = {
            'grasper': {
                'color_lower': np.array([0, 0, 200]),
                'color_upper': np.array([180, 50, 255]),
                'min_area': 0.02
            },
            'bipolar': {
                'threshold_type': cv2.THRESH_BINARY_INV,
                'min_area': 0.03
            },
            'hook': {
                'color_lower': np.array([20, 100, 100]),
                'color_upper': np.array([30, 255, 255]),
                'min_area': 0.015
            },
            'scissors': {
                'canny_threshold1': 50,
                'canny_threshold2': 150,
                'min_area': 0.025
            },
            'default': {
                'min_area': 0.01,
                'canny_threshold1': 30,
                'canny_threshold2': 100
            }
        }

    def _init_bbox_enhancement_params(self):
        # Prepara parámetros para la mejora de bounding boxes
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((7, 7), np.uint8)

    def normalize_tool_name(self, tool_name: str) -> str:
        # Normaliza nombres de herramientas con espacios o variaciones
        # Paso 1: Convertir a minúsculas y eliminnar espacios alrededor
        tool_name = tool_name.lower().strip()

        # Paso 2: Reemplazar guiones bajos por espacios
        tool_name = tool_name.replace('_', ' ')

        # Paso 3: Verificar si es una forma alternativa conocida
        if tool_name in self.alternative_names:
            tool_name = self.alternative_names[tool_name]

        # Paso 4: Unificar espacios múltiples
        tool_name = ' '.join(tool_name.split())

        return tool_name

    def unify_tool_name(self, tool_name: str) -> str:
        # Normaliza y unifica nombres de herramientas
        # Limpieza básica
        tool_name = str(tool_name).lower().strip().replace('_', ' ').replace('-', ' ')
        tool_name = ' '.join(tool_name.split()) # Elimina espacios múltiples

        # Verificar mapeo directo
        if tool_name in self.tool_mapping:
            return self.tool_mapping[tool_name]
        
        # Búsqueda aproximada para casos con pequeñas variaciones
        for original_name, unified_name in self.tool_mapping.items():
            if original_name.replace(' ', '') == tool_name.replace(' ', ''):
                return unified_name
        
        return 'other'
    
    def get_unified_class_id(self, tool_name: str) -> Optional[int]:
        # Obtiene el ID de clase unificada
        unified_name = self.unify_tool_name(tool_name)
        return self.unified_classes.index(unified_name) if unified_name in self.unified_classes else None
    
    def list_original_tool_names(self) -> List[str]:
        # Función para listar todas las herramientas originales mapeadas
        return sorted(set(self.tool_mapping.keys()))
    
    def save_unified_config(self, output_dir: Path):
        # Guarda la configuración unificada en formato YAML para YOLO
        config = {
            'path': str(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.unified_classes)},
            'nc': len(self.unified_classes)
        }

        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(config, f)

    def enhance_weak_supervision(self, frame: np.ndarray, tool_class: str) -> Optional[List[float]]:
        """
        Genera pseudo-bounding boxes mejorados para TODAS las herramientas quirúrgicas.
        Retorna: [x_center, y_center, width, height] normalizados (0-1)
        """
        if not self.enhance_bboxes:
            return [0.5, 0.5, 1.0, 1.0]  # Bbox de imagen completa

        # Asegurar imagen con 3 canales
        try:
            if frame is None:
                logging.warning("Frame es None.")
                return [0.5, 0.5, 1.0, 1.0]

            # Imagen en escala de grises
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Imagen con canal alfa (BGRA)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            elif frame.shape[2] != 3:
                logging.warning(f"Formato de imagen no soportado: {frame.shape}")
                return [0.5, 0.5, 1.0, 1.0]
        except Exception as e:
            logging.warning(f"No se pudo convertir el frame a BGR correctamente: {str(e)}")
            return [0.5, 0.5, 1.0, 1.0]
        
        # Configuración específica por herramienta
        tool_config = {
            'grasper': {
                'color_space': 'HSV',
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 50, 255]),
                'morph_ops': ['close'],
                'min_area': 0.02
            },
            'bipolar': {
                'color_space': 'GRAY',
                'threshold': [0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU],
                'morph_ops': ['open'],
                'min_area': 0.03
            },
            'hook': {
                'color_space': 'HSV',
                'lower': np.array([20, 100, 100]),
                'upper': np.array([30, 255, 255]),
                'morph_ops': ['close', 'dilate'],
                'min_area': 0.015
            },
            'scissors': {
                'color_space': 'EDGES',
                'canny': [50, 150],
                'morph_ops': ['dilate'],
                'min_area': 0.025
            },
            'clipper': {
                'color_space': 'HSV',
                'lower': np.array([0, 0, 150]),
                'upper': np.array([180, 50, 255]),
                'morph_ops': ['close'],
                'min_area': 0.02
            },
            'irrigator': {
                'color_space': 'BGR',
                'lower': np.array([200, 200, 200]),
                'upper': np.array([255, 255, 255]),
                'morph_ops': ['erode', 'dilate'],
                'min_area': 0.04
            },
            'specimen_bag': {
                'color_space': 'HSV',
                'lower': np.array([90, 50, 50]),
                'upper': np.array([120, 255, 255]),
                'morph_ops': ['close'],
                'min_area': 0.05
            },
            'other': {
                'color_space': 'EDGES',
                'canny': [30, 100],
                'morph_ops': ['dilate'],
                'min_area': 0.01
            }
        }.get(tool_class, {
            'color_space': 'EDGES',
            'canny': [30, 100],
            'morph_ops': ['dilate'],
            'min_area': 0.01
        })

        try:
            # Paso 1: Preprocesamiento
            if tool_config['color_space'] == 'HSV':
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, tool_config['lower'], tool_config['upper'])
                processed = mask
            elif tool_config['color_space'] == 'GRAY':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, processed = cv2.threshold(gray, *tool_config['threshold'])
            elif tool_config['color_space'] == 'BGR':
                mask = cv2.inRange(frame, tool_config['lower'], tool_config['upper'])
                processed = mask
            else:  # EDGES
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = cv2.Canny(gray, *tool_config['canny'])

            # Paso 2: Morfología
            for op in tool_config['morph_ops']:
                if op == 'close':
                    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, self.kernel_medium)
                elif op == 'open':
                    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, self.kernel_small)
                elif op == 'dilate':
                    processed = cv2.dilate(processed, self.kernel_medium, iterations=1)
                elif op == 'erode':
                    processed = cv2.erode(processed, self.kernel_small, iterations=1)

            # Paso 3: Contornos
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return [0.5, 0.5, 1.0, 1.0]

            # Paso 4: Filtro por área
            min_area = tool_config['min_area'] * frame.shape[0] * frame.shape[1]
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if not valid_contours:
                return [0.5, 0.5, 1.0, 1.0]

            # Paso 5: Bounding box
            main_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)

            # Paso 6: Formato YOLO
            height, width = frame.shape[:2]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            box_width = w / width
            box_height = h / height

            return [
                max(0.0, min(1.0, x_center)),
                max(0.0, min(1.0, y_center)),
                max(0.01, min(1.0, box_width)),
                max(0.01, min(1.0, box_height))
            ]

        except Exception as e:
            logging.warning(f"Error en enhance_weak_supervision para {tool_class}: {str(e)}")
            return [0.5, 0.5, 1.0, 1.0]