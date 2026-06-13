import numpy as np


class Kalman:
    def __init__(self, dt: float = 1.0):
        self._dt = dt
        self._state = np.zeros(4)
        self._P = np.eye(4) * 1000
        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self._Q = np.eye(4) * 0.1
        self._R = np.eye(2) * 1.0
        self._initialized = False

    def init(self, x: float, y: float):
        self._state = np.array([x, y, 0.0, 0.0])
        self._P = np.eye(4) * 1000
        self._initialized = True

    def reset(self):
        self._state = np.zeros(4)
        self._P = np.eye(4) * 1000
        self._initialized = False

    def update(self, x: float, y: float):
        if not self._initialized:
            self.init(x, y)
            return
        self._predict()
        z = np.array([x, y])
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ (z - self._H @ self._state)
        self._P = (np.eye(4) - K @ self._H) @ self._P

    def fuse(self) -> tuple[float, float, float, float]:
        return (
            self._state[0],
            self._state[1],
            self._state[2],
            self._state[3],
        )

    def _predict(self):
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
