import numpy as np
from third_party.bytetrack.byte_tracker import BYTETracker
from third_party.bytetrack.track import STrack

"""
Este módulo define una clase ByteTrackWrapper, una envoltura simplificada que integra el rastreador 
BYTETracker (de ByteTrack) para aplicar seguimiento multiobjeto sobre detecciones realizadas por modelos como YOLO.
"""

class ByteTrackWrapper:
    def __init__(self, track_thresh=0.3, match_thresh=0.8, buffer_size=90, frame_rate=30):
        args = {
            'track_thresh': track_thresh,
            'match_thresh': match_thresh,
            'track_buffer': buffer_size,
            'frame_rate': frame_rate,
        }
        self.tracker = BYTETracker(args, frame_rate)

    def update(self, bboxes_xyxy, scores, class_ids, frame):
        if len(bboxes_xyxy) == 0:
            self.tracker.update(np.empty((0, 5)), frame)
            return []

        # Concatenar bbox + score por ByteTrack formato
        detections = np.hstack((np.array(bboxes_xyxy), np.array(scores).reshape(-1, 1)))
        online_targets = self.tracker.update(detections, frame)

        # Asignar class_id si es posible (solo si el orden se mantiene)
        for i, t in enumerate(online_targets):
            if i < len(class_ids):
                t.cls = class_ids[i]

        results = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            track_id = t.track_id
            cls = int(t.cls) if hasattr(t, 'cls') else 0
            results.append([int(x1), int(y1), int(x2), int(y2), track_id, cls])

        return results
