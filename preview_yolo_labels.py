# preview_yolo_labels.py
import cv2
from pathlib import Path

IMG_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1tra in(626p)")
LABEL_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/labels_yolo")
OUT = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/yolo_preview")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["company","date","invoice","total","address"]
COLORS = [(0,255,0),(0,128,255),(0,0,255),(255,0,0),(255,255,0)]  # invoice is index 2 (blue)

for label_file in sorted(LABEL_DIR.glob("*.txt"))[:30]:   # preview up to 30
    stem = label_file.stem
    img_path = None
    for ext in [".jpg",".jpeg",".png"]:
        p = IMG_DIR / (stem + ext)
        if p.exists():
            img_path = p; break
    if img_path is None:
        continue
    img = cv2.imread(str(img_path))
    h,w = img.shape[:2]
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cid = int(parts[0]); xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        col = COLORS[cid % len(COLORS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        cv2.putText(img, CLASSES[cid], (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    out = OUT / (stem + "_preview.jpg")
    cv2.imwrite(str(out), img)
    print("Saved preview:", out)
