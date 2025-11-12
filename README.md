# 🧾 AI-Driven Multi-Stage Invoice Understanding and Validation System using Deep Learning

An AI-powered pipeline that automatically detects, reads, and validates invoice data using deep learning.  
Combines **YOLOv5** for field detection, **EasyOCR** for text extraction, and **rule-based validation** for data consistency and reliability.

## Project Overview

Manual invoice entry is slow and error-prone.  
This project automates the process through a **multi-stage deep learning pipeline**:

1. **Preprocessing** – Denoising, deskewing, resizing, and contrast enhancement.  
2. **YOLOv5 Field Detection** – Detects key invoice fields (Invoice No., Date, Vendor, Amount, etc.).  
3. **OCR (EasyOCR)** – Reads detected fields and extracts text with confidence scores.  
4. **Validation** – Performs rule-based consistency checks.  
5. **Structured Output** – Exports results in CSV/JSON format.

## ⚙️ Tech Stack

| Component | Tool / Library |
|------------|----------------|
| Language | Python 3.10 |
| Object Detection | YOLOv5 (Ultralytics) |
| OCR | EasyOCR |
| Preprocessing | OpenCV, NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab (T4 GPU) |

## System Architecture

Input Invoice → Preprocessing → YOLOv5 Detection → ROI Cropping → EasyOCR → Validation → Output

## 📂 Folder Structure

data/, models/, scripts/, outputs/, assets/

## 📊 Training Details

Dataset: SROIE 2019  
Epochs: 100 | Batch Size: 16 | GPU: T4  
Precision: 0.90 | Recall: 0.62 | mAP@0.5: 0.65 | mAP@0.5:0.95: 0.45

## 🧮 Preprocessing Steps
Deskewing, Denoising, Resizing, Contrast Enhancement (CLAHE).

## ✅ Validation Checks
Format & arithmetic checks with future fuzzy validation.

##  Installation

git clone https://github.com/<your-username>/AI-Invoice-Understanding.git
cd AI-Invoice-Understanding
pip install -r requirements.txt

## 🧑‍💻 Author
**Ninad Sarang**  
AI & Data Science Enthusiast | Deep Learning Researcher  


## 📜 License
MIT License
