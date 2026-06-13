import struct
import time

import serial as pyserial


class Serial:
    def __init__(self, port: str, baud_rate: int = 115200,
                 header: bytes = b'\xA5', send_freq: float = 50.0):
        self.baud_rate = baud_rate
        self.header = header
        self.send_freq = send_freq
        self._interval = 1.0 / send_freq
        self._ser = pyserial.Serial(port, baud_rate, timeout=0.1)
        self._last_send_time = 0.0

    def send(self, yaw: float, pitch: float):
        now = time.time()
        if now - self._last_send_time < self._interval:
            return
        data = self._build_frame(yaw, pitch)
        self._ser.write(data)
        self._last_send_time = now

    def _build_frame(self, yaw: float, pitch: float) -> bytes:
        return self.header + struct.pack('<ff', yaw, pitch)

    def close(self):
        if self._ser.is_open:
            self._ser.close()
