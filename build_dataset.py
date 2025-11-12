import os, glob, random, shutil

# paths
img_src = "/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/0325updated.task1train(626p)"  # Task1 images
labels_dir = "/Users/ninad/AuditFlow-AI/AuditFlow-AI/labels_yolo"                     # generated YOLO labels
root = "/Users/ninad/AuditFlow-AI/AuditFlow-AI/data/dataset_yolo"

# create dataset folder structure
os.makedirs(root + "/images/train", exist_ok=True)
os.makedirs(root + "/images/val", exist_ok=True)
os.makedirs(root + "/labels/train", exist_ok=True)
os.makedirs(root + "/labels/val", exist_ok=True)

# collect images that have labels
all_imgs = sorted(glob.glob(img_src + "/*.jpg"))
valid = []
for img in all_imgs:
    stem = os.path.splitext(os.path.basename(img))[0]
    if os.path.exists(os.path.join(labels_dir, stem + ".txt")):
        valid.append(img)

print("Total usable images:", len(valid))

# train/val split (80/20)
random.seed(42)
random.shuffle(valid)
val_n = max(1, int(0.2 * len(valid)))
val = valid[:val_n]
train = valid[val_n:]

def copy_files(files, split):
    for p in files:
        stem = os.path.basename(p)
        shutil.copy(p, root + f"/images/{split}/" + stem)
        shutil.copy(os.path.join(labels_dir, stem.replace(".jpg",".txt")), 
                    root + f"/labels/{split}/" + stem.replace(".jpg",".txt"))

copy_files(train, "train")
copy_files(val, "val")

print("Train images:", len(train), "Val images:", len(val))
