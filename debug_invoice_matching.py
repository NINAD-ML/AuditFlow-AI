# src/generate_yolo_labels_ngram.py
import os, re, json
from pathlib import Path
from rapidfuzz import fuzz
import cv2

# ----------------- USER PATHS -----------------
TASK1_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)")
TASK2_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task2train(626p)")
OUT = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/labels_yolo")
OUT.mkdir(parents=True, exist_ok=True)
LOG = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/labels_yolo_match_log.txt")

# ----------------- CONFIG -----------------
CLASSES = ["company", "date", "invoice", "total", "address"]  # invoice index 2
FUZZY_THRESHOLD = 78   # lower to 70 if you want more matches (may add false positives)
MAX_NGRAM = 4

# ----------------- helpers -----------------
def normalize(s):
    if s is None:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def parse_task1(txt_path: Path):
    """Return list of tokens: [{'text':..., 'bbox':(x1,y1,x2,y2)}]"""
    items = []
    s = txt_path.read_text(encoding="utf-8", errors="ignore")
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            parts = line.rsplit(",", 8)
            if len(parts) < 9:
                continue
        # first 8 numbers are coordinates
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

def ngrams(tokens, n):
    out = []
    for i in range(len(tokens)-n+1):
        seq = tokens[i:i+n]
        txt = " ".join([t["text"] for t in seq])
        bboxes = [t["bbox"] for t in seq]
        x1 = min(b[0] for b in bboxes); y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes); y2 = max(b[3] for b in bboxes)
        out.append({"text": txt, "bbox": (x1,y1,x2,y2), "start": i, "end": i+n-1})
    return out

# find best match among tokens + ngrams
def find_best_match(value, tokens):
    if not value:
        return None, 0, None  # matched_item, score, span_info

    target_norm = normalize(value)
    best = None
    best_score = -1
    best_span = None

    # 1) single token exact normalized substring
    for i, t in enumerate(tokens):
        tnorm = normalize(t["text"])
        if target_norm and target_norm in tnorm:
            return t, 100, {"type":"single","start":i,"end":i}

    # 2) ngrams exact normalized substring (prefer longer ngrams)
    for n in range(MAX_NGRAM,1,-1):
        for gram in ngrams(tokens, n):
            gnorm = normalize(gram["text"])
            if target_norm and target_norm in gnorm:
                return {"text": gram["text"], "bbox": gram["bbox"]}, 100, {"type":"ngram","start":gram["start"],"end":gram["end"],"n":n}

    # 3) fuzzy match (single tokens)
    for i, t in enumerate(tokens):
        tnorm = normalize(t["text"])
        if not tnorm: continue
        score = fuzz.partial_ratio(target_norm, tnorm)
        if score > best_score:
            best_score = score; best = t; best_span = {"type":"single","start":i,"end":i}

    # 4) fuzzy match on ngrams (2..MAX_NGRAM)
    for n in range(2, MAX_NGRAM+1):
        for gram in ngrams(tokens, n):
            gnorm = normalize(gram["text"])
            if not gnorm: continue
            score = fuzz.partial_ratio(target_norm, gnorm)
            if score > best_score:
                best_score = score
                best = {"text": gram["text"], "bbox": gram["bbox"]}
                best_span = {"type":"ngram","start":gram["start"],"end":gram["end"],"n":n}

    return best, best_score, best_span

# ----------------- main -----------------
def main():
    LOG.write_text("")  # reset log
    total_written = 0
    invoice_matches = 0

    for task2_txt in sorted(TASK2_DIR.glob("*.txt")):
        stem = task2_txt.stem
        task1_txt = TASK1_DIR / (stem + ".txt")
        # find image
        img_file = None
        for ext in [".jpg",".jpeg",".png"]:
            p = TASK1_DIR / (stem + ext)
            if p.exists():
                img_file = p; break
            p2 = TASK2_DIR / (stem + ext)
            if p2.exists():
                img_file = p2; break
        if not task1_txt.exists() or img_file is None:
            continue

        tokens = parse_task1(task1_txt)
        if not tokens:
            continue

        # parse task2 metadata
        s = task2_txt.read_text(encoding="utf-8", errors="ignore").strip()
        try:
            meta = json.loads(s)
            meta = {k.lower(): str(v).strip() for k,v in meta.items()}
        except:
            meta = {}
            for line in s.splitlines():
                if ":" in line:
                    k,v = line.split(":",1)
                    meta[k.strip().lower()] = v.strip().strip('", ')

        # load image dims
        img = cv2.imread(str(img_file))
        if img is None: 
            continue
        h,w = img.shape[:2]

        labels = []
        log_lines = []
        for cid, fname in enumerate(CLASSES):
            # find value in meta using variants
            val = None
            for key in [fname, fname + " no", fname + " no.", fname.replace("invoice","invoice no")]:
                if key in meta:
                    val = meta[key]; break
            if not val:
                # try any meta key containing fname
                for k in meta:
                    if fname in k:
                        val = meta[k]; break
            if not val:
                continue

            match, score, span = find_best_match(val, tokens)
            log_lines.append(f"FIELD={fname}  value='{val}'  best_score={score}  span={span}  match_text='{match['text'] if match and isinstance(match,dict) and 'text' in match else (match['text'] if match and 'text' in match else (match['text'] if match and 'text' in match else (match['text'] if match else None)))}'")

            if match and score >= FUZZY_THRESHOLD:
                # use match bbox
                bbox = match["bbox"] if isinstance(match, dict) and "bbox" in match else match["bbox"]
                x1,y1,x2,y2 = bbox
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                labels.append((cid, xc, yc, bw, bh))
                if fname == "invoice":
                    invoice_matches += 1

        # write labels file if any
        if labels:
            outf = OUT / (stem + ".txt")
            with open(outf, "w", encoding="utf-8") as f:
                for cid, xc, yc, bw, bh in labels:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            total_written += 1

        # append to log
        with open(LOG, "a", encoding="utf-8") as logf:
            logf.write(f"=== {stem} ===\n")
            for ll in log_lines:
                logf.write(ll + "\n")
            logf.write("\n")

    print(f"Done. Wrote labels for {total_written} images. Invoice matches: {invoice_matches}")
    print("Log written to:", LOG)

if __name__ == "__main__":
    main()
