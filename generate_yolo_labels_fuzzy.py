# generate_yolo_labels_fuzzy.py
# Put this in AuditFlow-AI/src/
import os, re, json
from pathlib import Path
from rapidfuzz import fuzz
import cv2

# ---------- USER-SPECIFIC PATHS (already set to your folders) ----------
# Task1 (word boxes + images)
TASK1_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)")
# Task2 (metadata .txt files with company,date,total,invoice etc.)
TASK2_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task2train(626p)")
# Output folder for YOLO labels
OUT = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/labels_yolo")
OUT.mkdir(parents=True, exist_ok=True)

print("TASK1_DIR:", TASK1_DIR)
print("TASK2_DIR:", TASK2_DIR)
print("OUT:", OUT)

# class mapping (order -> class_id)
CLASSES = ["company", "date", "invoice", "total", "address"]  # invoice = index 2

# fuzzy / normalization settings
FUZZY_THRESHOLD = 78   # start 78; lower to 70 if too few matches
def normalize(s):
    if s is None: return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# parse task1 word-level files (SROIE Task1 lines: 8 coords, then text)
def parse_task1_file(txt_path: Path):
    items = []
    s = txt_path.read_text(encoding="utf-8", errors="ignore")
    for line in s.splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split(",")
        if len(parts) < 9:
            parts = line.rsplit(",", 8)
            if len(parts) < 9:
                continue
        try:
            coords = list(map(int, parts[:8]))
        except:
            continue
        text = ",".join(parts[8:]).strip().strip('"')
        xs = coords[0::2]; ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        items.append({"text": text, "bbox": (x1,y1,x2,y2)})
    return items

# parse task2 metadata file (JSON-like or key:value)
def parse_task2_file(txt_path: Path):
    s = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        obj = json.loads(s)
        return {k.lower(): str(v).strip() for k,v in obj.items()}
    except:
        out = {}
        for line in s.splitlines():
            if ":" in line:
                k,v = line.split(":",1)
                out[k.strip().lower().strip('"')] = v.strip().strip('", ')
        return out

# find best matching token from task1 tokens for a given value
def find_best_match(value, tokens):
    if not value: return None, 0
    nv = normalize(value)
    best = None; best_score = 0
    for t in tokens:
        nt = normalize(t["text"])
        if not nt: continue
        if nv and nv in nt:
            return t, 100
        score = fuzz.partial_ratio(nv, nt)
        if score > best_score:
            best_score = score; best = t
    return best, best_score

# main loop
count_labels = 0
count_invoice_matched = 0
processed = 0

for task2_txt in sorted(TASK2_DIR.glob("*.txt")):
    stem = task2_txt.stem
    task1_txt = TASK1_DIR / (stem + ".txt")
    # find image in task1 dir (or task2 dir as fallback)
    img_file = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = TASK1_DIR / (stem + ext)
        if p.exists():
            img_file = p; break
        p2 = TASK2_DIR / (stem + ext)
        if p2.exists():
            img_file = p2; break
    if not task1_txt.exists() or img_file is None:
        # no matching task1 annotation or image found – skip
        continue

    tokens = parse_task1_file(task1_txt)
    fields = parse_task2_file(task2_txt)

    # read image for size
    img = cv2.imread(str(img_file))
    if img is None:
        continue
    h, w = img.shape[:2]

    labels = []
    for cid, fname in enumerate(CLASSES):
        val = None
        # try common keys and synonyms
        for key in [fname, fname.replace("invoice","invoice no"), fname + " no", fname + " no."]:
            if key in fields:
                val = fields[key]; break
        if not val:
            # try any field containing name
            for k in fields:
                if fname in k:
                    val = fields[k]; break
        if not val:
            continue

        match, score = find_best_match(val, tokens)
        if match and score >= FUZZY_THRESHOLD:
            x1,y1,x2,y2 = match["bbox"]
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            labels.append((cid, xc, yc, bw, bh))
            if fname == "invoice": count_invoice_matched += 1

    if labels:
        out_file = OUT / (stem + ".txt")
        with open(out_file, "w", encoding="utf-8") as f:
            for (cid,xc,yc,bw,bh) in labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        count_labels += 1

    processed += 1
    if processed % 50 == 0:
        print("Processed", processed, "task2 files...")

print("Done. Wrote labels for", count_labels, "images.")
print("Invoice-class matches:", count_invoice_matched)
print("Output labels folder:", OUT)
