def solve_yaw_pitch(
    cx: float,
    cy: float,
    frame_width: int = 640,
    frame_height: int = 480,
    fov_x: float = 60.0,
    fov_y: float = 45.0,
) -> tuple[float, float]:
    offset_x = cx - frame_width / 2
    offset_y = cy - frame_height / 2
    yaw = (offset_x / (frame_width / 2)) * (fov_x / 2)
    pitch = (offset_y / (frame_height / 2)) * (fov_y / 2)
    return yaw, pitch
