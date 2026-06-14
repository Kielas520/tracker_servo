from ultralytics import YOLO
import numpy as np


class YoloBackend:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.5):
        self._model = YOLO(model_path)
        self._conf = conf
        self._target_class: str | None = None

    def set_classes(self, classes: list[str]):
        pass

    def detect_all(self, frame: np.ndarray) -> list[dict]:
        results = self._model(frame, conf=self._conf, verbose=False)
        targets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                targets.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class": cls_name,
                    "conf": conf,
                })
        return targets

    def set_target_class(self, cls_name: str):
        self._target_class = cls_name

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        targets = self.detect_all(frame)
        if self._target_class is None:
            return None
        for t in targets:
            if t["class"] == self._target_class:
                x1, y1, x2, y2 = t["bbox"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                return (cx, cy)
        return None

    def get_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        targets = self.detect_all(frame)
        if self._target_class is None:
            return None
        for t in targets:
            if t["class"] == self._target_class:
                return t["bbox"]
        return None
