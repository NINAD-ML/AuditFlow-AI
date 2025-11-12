import os
import cv2
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_DIR = "/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task2train(626p)"   # path to Task2 dataset
OUTPUT_DIR = "outputs/visualized"                # folder for saving visualizations

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class map for readability (adjust if you add more fields)
CLASS_MAP = {
    0: "Company",
    1: "Date",
    2: "Address",
    3: "Total"
}

def draw_boxes(image_path, label_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        return

    h, w, _ = image.shape

    # Read YOLO-style label file
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip bad lines

        class_id, x_center, y_center, box_w, box_h = map(float, parts)
        class_id = int(class_id)

        # Convert normalized coords back to pixel values
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = CLASS_MAP.get(class_id, str(class_id))
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)

    # Save visualization
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            img_path = os.path.join(DATA_DIR, file)
            txt_path = os.path.join(DATA_DIR, file.replace(".jpg", ".txt"))

            if os.path.exists(txt_path):
                out_path = os.path.join(OUTPUT_DIR, file)
                draw_boxes(img_path, txt_path, out_path)
                print(f"✅ Processed {file}")
            else:
                print(f"⚠️ No annotation found for {file}")
