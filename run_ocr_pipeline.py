import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import easyocr
import re

# --- Paths (adjust if needed) ---
weights = "weights/best.pt"         # Trained YOLO model
source = "data/test_invoices"       # Folder with test invoice images
out_dir = Path("results_ocr")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load YOLO model ---
print("🔹 Loading YOLO model...")
model = YOLO(weights)

# --- Load EasyOCR ---
print("🔹 Loading EasyOCR...")
reader = easyocr.Reader(['en'])

# --- Records for CSV ---
records = []

# Regex helpers for cleaning OCR text
def clean_invoice_number(text):
    match = re.search(r'([A-Z]*\d{4,})', text.replace(" ", ""))
    return match.group(1) if match else text.strip()

def clean_date(text):
    match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
    return match.group(1) if match else text.strip()

def clean_total(text):
    match = re.search(r'(\d+\.\d{2})', text.replace(",", ""))
    return match.group(1) if match else text.strip()

# --- Process each invoice ---
for img_path in Path(source).glob("*.jpg"):
    print(f"📄 Processing {img_path.name} ...")
    
    # Run YOLO detection
    results = model(str(img_path))
    detections = results[0].boxes.data.cpu().numpy()  # xyxy + conf + cls
    
    img = cv2.imread(str(img_path))
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        label = model.names[cls_id]   # class name (e.g., invoice, company, total, date)
        
        # Crop the detected box
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        
        # OCR
        ocr_result = reader.readtext(crop)
        text = " ".join([t[1] for t in ocr_result]) if ocr_result else ""
        
        # Clean based on field
        if label.lower() == "invoice":
            text = clean_invoice_number(text)
        elif label.lower() == "date":
            text = clean_date(text)
        elif label.lower() == "total":
            text = clean_total(text)
        
        records.append({
            "image": img_path.name,
            "field": label,
            "text": text,
            "confidence": float(conf)
        })

# --- Save to CSV ---
df = pd.DataFrame(records)
out_csv = out_dir / "invoice_extracted.csv"
df.to_csv(out_csv, index=False)

print(f"\n✅ Done! Extracted invoice fields saved to {out_csv}")
