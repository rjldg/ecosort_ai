# apps/realtime_trash_detector.py
"""
Real-time trash detection from a laptop webcam using OpenCV + Ultralytics YOLOv11.
- Uses cv2.VideoCapture for explicit webcam control.
- Draws model overlays via Result.plot() for correctness (boxes/labels/masks).
- Shows FPS and saves snapshots on keypress.

Keys:
  q = quit
  s = save annotated frame to ./runs/realtime

Why: Reliable overlay rendering and simple, readable control flow.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from ultralytics import YOLO


def run_webcam(
    weights: str = "runs/detect/train7/weights/best.pt",
    camera: int = 0,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | int = 0,
    window_name: str = "Trash Detector (YOLOv11)",
    save_dir: str | Path = "runs/realtime",
    autofocus: bool = False,
) -> None:
    """Start webcam loop and run YOLO inference per frame.

    Notes:
        - Uses result.plot() to avoid mismatched color spaces or missing overlays.
        - `device` accepts CUDA index (0) or 'cpu'.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {camera}")

    # Optional autofocus toggle (not supported on all cams)
    if autofocus:
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_tick = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from webcam.")
                break

            # Inference (single frame). "device" here steers CUDA/CPU.
            results = model(
                frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )
            r = results[0]

            annotated = r.plot()  # BGR with overlays

            # FPS calculation (simple moving estimate)
            frame_count += 1
            now = time.time()
            dt = now - last_tick
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                last_tick = now

            # Draw HUD text
            hud = f"FPS: {fps:4.1f}  |  det: {len(r.boxes) if r.boxes is not None else 0}"
            cv2.putText(
                annotated,
                hud,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                hud,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = int(time.time() * 1000)
                out_file = save_path / f"frame_{ts}.jpg"
                # Use Ultralytics save to ensure overlay parity
                try:
                    r.save(filename=str(out_file))
                except Exception:
                    cv2.imwrite(str(out_file), annotated)
                print(f"[INFO] Saved {out_file}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Adjust `weights` to your trained model path as needed.
    run_webcam(
        weights="runs/detect/train7/weights/best.pt",
        camera=0,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=0,  # set to 'cpu' if no GPU
        window_name="Trash Detector (YOLOv11)",
        save_dir="runs/realtime",
        autofocus=False,
    )
