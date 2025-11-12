# src/visualize_task1.py
import cv2, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# UPDATE this to your Task-1 folder and a sample file that exists
TASK1_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)")
SAMPLE = next(TASK1_DIR.glob("*.jpg"))

def draw_one(img_path):
    txt_path = img_path.with_suffix(".txt")
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    if not txt_path.exists():
        raise RuntimeError(f"Missing annotation: {txt_path}")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            # first 8 are coords (quad), rest is text (may contain commas)
            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue
            text = ",".join(parts[8:]).strip().strip('"')

            # draw polygon
            pts = np.array([[coords[0],coords[1]],
                            [coords[2],coords[3]],
                            [coords[4],coords[5]],
                            [coords[6],coords[7]]], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)

            # label near the top-left of the quad
            tlx, tly = pts.min(axis=0)
            cv2.putText(img, text[:20], (int(tlx), int(max(12, tly-5))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,12))
    plt.imshow(img_rgb); plt.axis("off")
    plt.tight_layout()
    out = img_path.with_name(img_path.stem + "_annotated.jpg")
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    if SAMPLE:
        draw_one(SAMPLE)
    else:
        print("No .jpg found in", TASK1_DIR)

