import cv2
import numpy as np


class Display:
    def __init__(self, window_name: str = "Tracker Servo"):
        self._window_name = window_name
        self.enabled: bool = True

    def show(self, img: np.ndarray | None):
        if not self.enabled or img is None:
            return
        cv2.imshow(self._window_name, img)

    def wait_key(self, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def destroy(self):
        cv2.destroyAllWindows()
