import struct
import threading
import time
from pathlib import Path

import yaml
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.prompt import IntPrompt

from cam import Camera
from detector import Detector
from tracker import Tracker
from kie_serial import Serial
from ui import Display

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()


class FpsCounter:
    def __init__(self):
        self._count = 0
        self._last_time = time.time()
        self.fps = 0.0

    def tick(self):
        self._count += 1
        now = time.time()
        dt = now - self._last_time
        if dt >= 0.5:
            self.fps = self._count / dt
            self._count = 0
            self._last_time = now


def build_serial_hex(header: bytes, yaw: float, pitch: float) -> str:
    frame = header + struct.pack('<ff', yaw, pitch)
    return ' '.join(f'{b:02X}' for b in frame)


def build_status_table(state):
    table = Table(expand=True, show_header=False, box=None, padding=(0, 1))
    table.add_column("L", style="cyan", width=14)
    table.add_column("R", style="white")

    det_center = state.get("raw_center")
    det_str = f"({det_center[0]:.1f}, {det_center[1]:.1f})" if det_center else "-"

    filt = state.get("filtered_state")
    if filt:
        filt_pos = f"({filt[0]:.1f}, {filt[1]:.1f})"
        filt_vel = f"({filt[2]:.1f}, {filt[3]:.1f})"
    else:
        filt_pos = "-"
        filt_vel = "-"

    yaw, pitch = state.get("yaw", 0.0), state.get("pitch", 0.0)
    header = bytes.fromhex(CFG["serial"]["header"])
    hex_str = build_serial_hex(header, yaw, pitch)
    frame_len = len(header) + 8

    ser_cfg = CFG["serial"]
    serial_status = "[green]CONNECTED[/green]" if ser_cfg["enabled"] else "[dim]DISABLED[/dim]"
    serial_port = f"{ser_cfg['port']} @ {ser_cfg['baud_rate']}" if ser_cfg["enabled"] else "-"

    table.add_row("[bold]Detector[/bold]", f"{state['det_fps']:.1f} Hz  raw={det_str}")
    table.add_row("[bold]Kalman[/bold]", f"{state['trk_fps']:.1f} Hz  pos={filt_pos}  vel={filt_vel}")
    table.add_row("[bold]Output[/bold]", f"yaw=[yellow]{yaw:+.2f}°[/yellow]  pitch=[yellow]{pitch:+.2f}°[/yellow]")
    table.add_row("", "")
    table.add_row("[bold]Serial[/bold]", f"{serial_status}  {state['ser_fps']:.1f} Hz")
    table.add_row("  Port", serial_port)
    table.add_row("  Frame", f"[green]{hex_str}[/green]")
    table.add_row("  Struct", f"[dim]HDR[{len(header)}B] YAW[4B] PIT[4B] = {frame_len}B[/dim]")

    return Panel(table, title="[bold]Tracker Servo[/bold]", border_style="blue")


def select_target_yolo(detector, frame):
    targets = detector.warm_up(frame)
    if not targets:
        console.print("[red]No targets detected.[/red]")
        return False
    console.print("\n[bold]--- Detected Targets ---[/bold]")
    for i, t in enumerate(targets):
        console.print(f"  [cyan][{i}][/cyan] {t['class']} (conf: {t['conf']:.2f})")
    console.print("[bold]------------------------[/bold]")
    idx = IntPrompt.ask("Select target index")
    if 0 <= idx < len(targets):
        detector.confirm_target(targets[idx])
        return True
    return False


def tracking_loop(cam, detector, tracker_obj, ser, state):
    hz = CFG["tracker"]["hz"]
    interval = 1.0 / hz
    det_fps = FpsCounter()
    trk_fps = FpsCounter()
    ser_fps = FpsCounter()

    while state["running"]:
        t0 = time.time()

        frame = cam.read()

        center = detector.detect(frame)
        det_fps.tick()
        state["det_fps"] = det_fps.fps
        state["raw_center"] = center

        result = tracker_obj.update(center)
        trk_fps.tick()
        state["trk_fps"] = trk_fps.fps
        state["filtered_state"] = tracker_obj.filtered_state

        if result is not None:
            state["yaw"], state["pitch"] = result
            if ser:
                ser.send(state["yaw"], state["pitch"])
                ser_fps.tick()
        state["ser_fps"] = ser_fps.fps

        tracker_obj.draw(detector.draw_enabled, detector.img_draw)

        elapsed = time.time() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    cam_cfg = CFG["camera"]
    det_cfg = CFG["detector"]
    trk_cfg = CFG["tracker"]
    ser_cfg = CFG["serial"]

    cam = Camera(
        source=cam_cfg["source"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
    )
    detector = Detector(
        mode=det_cfg["mode"],
        yolo_model_path=str(ROOT / det_cfg["yolo_model_path"]),
        mix_model_path=str(ROOT / det_cfg["mix_model_path"]),
        classes=det_cfg["classes"],
        conf=det_cfg["conf"],
    )
    detector.draw_enabled = det_cfg["draw"]

    hz = trk_cfg["hz"]
    tracker_obj = Tracker(
        frame_width=cam.width,
        frame_height=cam.height,
        fov_x=trk_cfg["fov_x"],
        fov_y=trk_cfg["fov_y"],
        dt=1.0 / hz,
        max_lost=trk_cfg["max_lost"],
    )
    display = Display()

    ser = None
    if ser_cfg["enabled"]:
        header = bytes.fromhex(ser_cfg["header"])
        ser = Serial(
            port=ser_cfg["port"],
            baud_rate=ser_cfg["baud_rate"],
            header=header,
            send_freq=hz,
        )

    for _ in range(30):
        frame = cam.read()

    if detector.mode == "yolo":
        if not select_target_yolo(detector, frame):
            console.print("[red]Target selection failed.[/red]")
            cam.release()
            return
    else:
        detector.warm_up(frame)

    state = {
        "running": True,
        "yaw": 0.0,
        "pitch": 0.0,
        "raw_center": None,
        "filtered_state": None,
        "det_fps": 0.0,
        "trk_fps": 0.0,
        "ser_fps": 0.0,
    }

    track_thread = threading.Thread(
        target=tracking_loop,
        args=(cam, detector, tracker_obj, ser, state),
        daemon=True,
    )
    track_thread.start()

    with Live(build_status_table(state), refresh_per_second=10, console=console) as live:
        while state["running"]:
            if detector.draw_enabled and detector.img_draw is not None:
                display.show(detector.img_draw)
            key = display.wait_key(1)

            if key == ord('q'):
                state["running"] = False
            elif key == ord('d'):
                detector.draw_enabled = not detector.draw_enabled

            live.update(build_status_table(state))

    track_thread.join(timeout=1.0)
    cam.release()
    if ser:
        ser.close()
    display.destroy()


if __name__ == "__main__":
    main()
