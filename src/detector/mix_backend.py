import cv2
import numpy as np
import onnxruntime as ort


class MixBackend:
    def __init__(self, model_path: str, search_size: int = 224, template_size: int = 112,
                 search_factor: float = 4.5, template_factor: float = 2.0,
                 size_change_limit: float = 0.1):
        self._model_path = model_path
        self._session = None
        self._search_size = search_size
        self._template_size = template_size
        self._search_factor = search_factor
        self._template_factor = template_factor
        self._size_change_limit = size_change_limit
        self._target_pos = None
        self._target_sz = None
        self._init_sz = None
        self._template_patch = None
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self):
        if self._session is None:
            if not self._model_path:
                raise ValueError("MixBackend: model_path is not set. Provide a valid .onnx file path.")
            self._session = ort.InferenceSession(self._model_path)

    def init_tracker(self, frame: np.ndarray, bbox: tuple[int, int, int, int]):
        self._load_model()
        x1, y1, x2, y2 = bbox
        self._target_pos = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        self._target_sz = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        self._init_sz = self._target_sz.copy()
        self._template_patch = self._crop_patch(
            frame, self._target_pos, self._target_sz,
            self._template_factor, self._template_size
        )

    def track(self, frame: np.ndarray) -> tuple[float, float] | None:
        if self._target_pos is None or self._session is None:
            return None

        search_patch = self._crop_patch(
            frame, self._target_pos, self._target_sz,
            self._search_factor, self._search_size
        )

        outputs = self._session.run(None, {
            "template": self._template_patch,
            "search": search_patch,
        })

        pred_bbox = outputs[0][0]
        cx_norm, cy_norm, w_norm, h_norm = pred_bbox

        crop_sz = max(self._target_sz) * self._search_factor
        cx = cx_norm * crop_sz + (self._target_pos[0] - crop_sz / 2)
        cy = cy_norm * crop_sz + (self._target_pos[1] - crop_sz / 2)
        w = w_norm * crop_sz
        h = h_norm * crop_sz

        w = np.clip(w, self._init_sz[0] * 0.5, self._init_sz[0] * 3.0)
        h = np.clip(h, self._init_sz[1] * 0.5, self._init_sz[1] * 3.0)

        prev_w, prev_h = self._target_sz
        w = prev_w + np.clip(w - prev_w, -prev_w * self._size_change_limit, prev_w * self._size_change_limit)
        h = prev_h + np.clip(h - prev_h, -prev_h * self._size_change_limit, prev_h * self._size_change_limit)

        self._target_pos = np.array([cx, cy])
        self._target_sz = np.array([w, h])
        return (float(cx), float(cy))

    def get_bbox(self) -> tuple[int, int, int, int] | None:
        if self._target_pos is None or self._target_sz is None:
            return None
        x1 = int(self._target_pos[0] - self._target_sz[0] / 2)
        y1 = int(self._target_pos[1] - self._target_sz[1] / 2)
        x2 = int(self._target_pos[0] + self._target_sz[0] / 2)
        y2 = int(self._target_pos[1] + self._target_sz[1] / 2)
        return (x1, y1, x2, y2)

    def _crop_patch(self, frame: np.ndarray, pos: np.ndarray, target_sz: np.ndarray,
                    factor: float, output_size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        crop_sz = int(max(target_sz) * factor)
        cx, cy = pos
        x1 = int(cx - crop_sz / 2)
        y1 = int(cy - crop_sz / 2)
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        patch = frame[y1:y2, x1:x2].copy()
        if pad_left or pad_top or pad_right or pad_bottom:
            patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

        patch = cv2.resize(patch, (output_size, output_size))
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - self._mean) / self._std
        patch = np.transpose(patch, (2, 0, 1))[np.newaxis, ...]
        return patch
