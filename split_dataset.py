import os
import random
import shutil

SRC = "dataset/collected_raw"

IMG_EXT = ".jpg"

files = [f for f in os.listdir(SRC) if f.endswith(IMG_EXT)]
files = [f for f in files if os.path.exists(os.path.join(SRC, f.replace(".jpg", ".txt")))]

random.shuffle(files)

split = int(len(files) * 0.8)
train_files = files[:split]
val_files = files[split:]

def copy(files, mode):
    for f in files:
        img_src = os.path.join(SRC, f)
        lbl_src = os.path.join(SRC, f.replace(".jpg", ".txt"))

        img_dst = f"dataset/images/{mode}/{f}"
        lbl_dst = f"dataset/labels/{mode}/{f.replace('.jpg','.txt')}"

        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)

copy(train_files, "train")
copy(val_files, "val")

print("DONE:", len(train_files), "train /", len(val_files), "val")