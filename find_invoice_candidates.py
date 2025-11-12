import re
from pathlib import Path

TASK1_DIR = Path("/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)")

# Common regex patterns for invoice/document/bill numbers
PATTERNS = [
    r"(invoice\s*no\.?\s*[:\-]?\s*\w+)",   # Invoice No: XYZ123
    r"(doc(ument)?\s*no\.?\s*[:\-]?\s*\w+)",  # Doc No: TD01167104
    r"(bill\s*#?\s*\w+)",                  # Bill# V001-540835
    r"\b[A-Z]{1,3}\d{5,}\b",               # e.g., TD01167104, R000027830
    r"\bV\d{3,}-\d+\b",                    # e.g., V001-540835
    r"\b[A-Z]{2}\s?\d+\b",                 # e.g., CS 10012
    r"\b\d{8,}\b"                          # long numeric IDs e.g., 050100035279
]

def extract_candidates(txt_file):
    text = txt_file.read_text(encoding="utf-8", errors="ignore")
    candidates = []
    for pattern in PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            # Handle tuple returns (from grouped regexes)
            if isinstance(m, tuple):
                candidates.append(m[0])
            else:
                candidates.append(m)
    return list(set(candidates))  # unique

def main():
    for i, txt_file in enumerate(TASK1_DIR.glob("*.txt")):
        if i >= 20:  # limit preview to first 20 files for now
            break
        candidates = extract_candidates(txt_file)
        if candidates:
            print(f"{txt_file.stem}: {candidates}")

if __name__ == "__main__":
    main()
