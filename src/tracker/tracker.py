import cv2
import numpy as np

from .kalman import Kalman
from .geometry import solve_yaw_pitch


class Tracker:
    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 fov_x: float = 60.0, fov_y: float = 45.0,
                 dt: float = 1.0, max_lost: int = 10):
        self._kalman = Kalman(dt=dt)
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._fov_x = fov_x
        self._fov_y = fov_y
        self._max_lost = max_lost
        self._lost_count = 0
        self._tracking = False
        self.filtered_pos: tuple[float, float] | None = None
        self.filtered_state: tuple[float, float, float, float] | None = None
        self.yaw: float = 0.0
        self.pitch: float = 0.0

    def update(self, center: tuple[float, float] | None) -> tuple[float, float] | None:
        if center is None:
            self._lost_count += 1
            if self._lost_count > self._max_lost:
                self.reset()
                return None
            if self._tracking:
                self._kalman._predict()
                state = self._kalman.fuse()
                self.filtered_state = state
                self.filtered_pos = (state[0], state[1])
                self.yaw, self.pitch = solve_yaw_pitch(
                    state[0], state[1], self._frame_width, self._frame_height, self._fov_x, self._fov_y
                )
                return self.yaw, self.pitch
            return None

        if not self._tracking:
            self._kalman.init(center[0], center[1])
            self._tracking = True
        else:
            self._kalman.update(center[0], center[1])

        self._lost_count = 0
        state = self._kalman.fuse()
        self.filtered_state = state
        self.filtered_pos = (state[0], state[1])
        self.yaw, self.pitch = solve_yaw_pitch(
            state[0], state[1], self._frame_width, self._frame_height, self._fov_x, self._fov_y
        )
        return self.yaw, self.pitch

    def draw(self, draw_enabled: bool, img: np.ndarray | None) -> np.ndarray | None:
        if not draw_enabled or img is None or self.filtered_pos is None:
            return img
        cx, cy = int(self.filtered_pos[0]), int(self.filtered_pos[1])
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        return img

    def reset(self):
        self._kalman.reset()
        self._lost_count = 0
        self._tracking = False
        self.filtered_pos = None
        self.filtered_state = None
