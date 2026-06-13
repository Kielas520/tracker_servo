import cv2
import numpy as np

from .yolo_backend import YoloBackend
from .mix_backend import MixBackend


class Detector:
    def __init__(self, mode: str = "yolo",
                 yolo_model_path: str = "yolov8s-worldv2.pt",
                 mix_model_path: str = "",
                 classes: list[str] | None = None, conf: float = 0.5):
        self.mode = mode
        self.draw_enabled: bool = True
        self.img_raw: np.ndarray | None = None
        self.img_draw: np.ndarray | None = None
        self._target_info = None
        self._yolo_backend = None
        self._mix_backend = None

        if mode == "yolo":
            self._yolo_backend = YoloBackend(model_path=yolo_model_path, conf=conf)
            if classes:
                self._yolo_backend.set_classes(classes)
        elif mode == "mix":
            self._mix_backend = MixBackend(model_path=mix_model_path)

    def warm_up(self, frame: np.ndarray) -> list[dict] | None:
        self.img_raw = frame.copy()
        if self.mode == "yolo":
            return self._warm_up_yolo(frame)
        elif self.mode == "mix":
            self._warm_up_mix(frame)
            return None

    def _warm_up_yolo(self, frame: np.ndarray) -> list[dict]:
        targets = self._yolo_backend.detect_all(frame)
        if self.draw_enabled:
            self.img_draw = frame.copy()
            for i, t in enumerate(targets):
                x1, y1, x2, y2 = t["bbox"]
                cv2.rectangle(self.img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"[{i}] {t['class']} {t['conf']:.2f}"
                cv2.putText(self.img_draw, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return targets

    def _warm_up_mix(self, frame: np.ndarray):
        roi = cv2.selectROI("Select Target", frame, showCrosshair=True)
        cv2.destroyWindow("Select Target")
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            bbox = (x, y, x + w, y + h)
            self.confirm_target(bbox)

    def confirm_target(self, target):
        self._target_info = target
        if self.mode == "yolo" and isinstance(target, dict):
            self._yolo_backend.set_target_class(target["class"])
        elif self.mode == "mix":
            self._mix_backend.init_tracker(self.img_raw, target)

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        self.img_raw = frame.copy()
        center = None

        if self.mode == "yolo":
            center = self._yolo_backend.detect(frame)
        elif self.mode == "mix":
            center = self._mix_backend.track(frame)

        if self.draw_enabled:
            self._draw(frame, center)

        return center

    def _draw(self, frame: np.ndarray, center: tuple[float, float] | None):
        self.img_draw = frame.copy()

        if self.mode == "yolo":
            bbox = self._yolo_backend.get_bbox(self.img_raw)
        else:
            bbox = self._mix_backend.get_bbox()

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(self.img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(self.img_draw, (cx, cy), 4, (0, 255, 0), -1)
