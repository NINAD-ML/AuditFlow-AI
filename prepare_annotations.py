import os
import cv2
import matplotlib.pyplot as plt

# Path to your dataset
data_path = "/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task2train(626p)"
img_file = "X00016469612.jpg"
txt_file = "X00016469612.txt"

# Load image
img_path = os.path.join(data_path, img_file)
txt_path = os.path.join(data_path, txt_file)
img = cv2.imread(img_path)

# Read annotations
bboxes = []
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        # split only last 4 commas → [text, x1, y1, x2, y2]
        parts = line.strip().rsplit(",", 4)
        if len(parts) == 5:
            text = parts[0].strip()
            try:
                x1, y1, x2, y2 = map(int, parts[1:])
                bboxes.append((text, x1, y1, x2, y2))
            except ValueError:
                print("Skipping bad line:", line)

# Draw boxes on image
for text, x1, y1, x2, y2 in bboxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

