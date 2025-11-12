import os, json
from pathlib import Path

TASK1_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)")  # word boxes
TASK2_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task2train(626p)")  # key fields
OUT_DIR   = Path("data/labels_yolo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Classes we care about
CLASSES = ["company", "date", "invoice", "total", "address"]


def parse_task1(txt_path):
    """Return list of word boxes [(text, [x1,y1,x2,y2,x3,y3,x4,y4])]."""
    boxes = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9: continue
            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue
            text = ",".join(parts[8:]).strip().strip('"')
            boxes.append((text, coords))
    return boxes

def parse_task2(txt_path):
    """Return dict of key fields from Task-2 file."""
    try:
        data = json.loads(txt_path.read_text(encoding="utf-8"))
        return {k.lower(): str(v).strip() for k,v in data.items()}
    except:
        # fallback: key: value lines
        result = {}
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k,v = line.split(":",1)
                result[k.strip().lower()] = v.strip()
        return result

def coords_to_yolo(coords, img_w, img_h):
    """Convert quadrilateral coords to YOLO (x_center,y_center,w,h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h

def main():
    for img_path in TASK1_DIR.glob("*.jpg"):
        txt1 = img_path.with_suffix(".txt")
        txt2 = TASK2_DIR / (img_path.stem + ".txt")
        if not txt1.exists() or not txt2.exists():
            continue
        
        boxes = parse_task1(txt1)
        fields = parse_task2(txt2)

        # Load image size
        import cv2
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        label_lines = []
        for cid, cname in enumerate(CLASSES):
            if cname not in fields: continue
            val = fields[cname].lower()
            # find match in task1 boxes
            for text, coords in boxes:
                if val in text.lower():  # simple contains match
                    xc,yc,ww,hh = coords_to_yolo(coords, w, h)
                    label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
                    break
        
        if label_lines:
            out_file = OUT_DIR / (img_path.stem + ".txt")
            out_file.write_text("\n".join(label_lines))
            print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
