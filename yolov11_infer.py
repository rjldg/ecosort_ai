# notebooks/yolo11_predict_display.py
"""
Notebook cell to run predictions with a trained Ultralytics YOLOv11 model and display annotated results.
- Uses model's .plot() for boxes/masks and shows via matplotlib.
- Saves annotated frames to disk.
- Prints a concise table of detections.

Why: keep a single, reusable entrypoint for images/URLs/directories/videos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import matplotlib.pyplot as plt

# Optional: only needed if you prefer manual OpenCV saving; matplotlib save works too
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from ultralytics import YOLO

# -------------------------------
# 1) Load your trained model
# -------------------------------
# If you've already created `trained_model` earlier in the notebook, you can skip this line
MODEL_WEIGHTS = "runs/detect/train7/weights/best.pt"  # adjust if your run name differs
model = YOLO(MODEL_WEIGHTS)

# -------------------------------
# 2) Helper: show image (RGB np.ndarray) inline
# -------------------------------
def _imshow_rgb(img: np.ndarray, title: str | None = None) -> None:
    if img.ndim == 2:
        disp = img
    else:
        # Ensure RGB for matplotlib (Ultralytics returns BGR from .plot())
        disp = img[:, :, ::-1] if img.shape[-1] == 3 else img
    plt.figure(figsize=(10, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(disp)
    plt.show()


# -------------------------------
# 3) Core runner: predict + display + save
# -------------------------------
SourceLike = Union[str, Path, int, np.ndarray, List[Union[str, Path]]]

def predict_and_show(
    source: SourceLike,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Union[str, int] = 0,
    save_dir: Union[str, Path] = "runs/predict_display",
    show_inline: bool = True,
    save_images: bool = True,
) -> None:
    """Run inference and visualize results.

    Args:
        source: Image path/URL, directory, video path, webcam index, ndarray, or list of paths.
        imgsz: Inference size.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        device: CUDA index or 'cpu'.
        save_dir: Where annotated images/frames will be saved.
        show_inline: Display annotated image(s) inline (matplotlib).
        save_images: Save annotated outputs to files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = model(
        source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        stream=False,  # set True for very large batches/streams
    )

    # Ultralytics returns a list-like of Result objects
    for idx, r in enumerate(results):
        # r.plot() returns an annotated BGR image (np.ndarray)
        annotated_bgr = r.plot()  # handles boxes/masks/keypoints if present

        # Compose a readable title and filename
        orig_name = Path(r.path).name if hasattr(r, "path") and r.path else f"frame_{idx:06d}.jpg"
        title = f"{orig_name}  |  {len(r.boxes) if r.boxes is not None else 0} detections"
        out_path = save_dir / orig_name

        # Save annotated image
        if save_images:
            if _HAS_CV2:
                cv2.imwrite(str(out_path), annotated_bgr)
            else:
                # Fall back to matplotlib save (expects RGB)
                plt.imsave(str(out_path), annotated_bgr[:, :, ::-1])

        # Inline display
        if show_inline:
            _imshow_rgb(annotated_bgr, title=title)

        # Compact, per-image summary
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            names = [r.names[int(c)] for c in cls]
            # Tabulate
            print(f"\nDetections for: {orig_name}")
            print("idx  class          conf    x1     y1     x2     y2")
            for i, (n, p, box) in enumerate(zip(names, confs, xyxy)):
                x1, y1, x2, y2 = box
                print(f"{i:>3}  {n:<13}  {p:>5.2f}  {x1:>5.0f}  {y1:>5.0f}  {x2:>5.0f}  {y2:>5.0f}")
        else:
            print(f"\nNo detections for: {orig_name}")

        # Optional: show mask stats when available
        if getattr(r, "masks", None) is not None:
            n_masks = len(r.masks)
            print(f"Masks: {n_masks} instance(s)")


# -------------------------------
# 4) Example usage with your URL (runs inline)
# -------------------------------
if __name__ == "__main__":
    test_source = "https://cf.shopee.ph/file/e50eba5b2aa1a1cf32d0cb6b39630ce1"  # Example URL
    predict_and_show(
        test_source,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=0,  # set to 'cpu' if no GPU
        save_dir="runs/predict_display",
        show_inline=True,
        save_images=True,
    )


# -------------------------------
# 5) (Optional) Manual draw for full control
# -------------------------------
def manual_draw_boxes(
    result,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a manually annotated BGR image from a single Ultralytics Result.
    Why: gives you control over styling beyond .plot().
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for manual drawing.")

    img = result.orig_img.copy()  # BGR
    if result.boxes is None or len(result.boxes) == 0:
        return img

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.int().cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    names = [result.names[int(c)] for c in cls]

    for (x1, y1, x2, y2), name, conf in zip(xyxy, names, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{name} {conf:.2f}"
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img

# Example (uncomment to use after a call to model):
# res = model(test_source, imgsz=640, conf=0.25)[0]
# custom_bgr = manual_draw_boxes(res)
# _imshow_rgb(custom_bgr, title="Manual draw")
