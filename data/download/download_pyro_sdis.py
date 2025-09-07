#!/usr/bin/env python3
import os
from datasets import load_dataset
OUT = "code/data/pyro_sdis_yolo"
os.makedirs(f"{OUT}/images/train", exist_ok=True); os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True); os.makedirs(f"{OUT}/labels/val", exist_ok=True)
ds = load_dataset("pyronear/pyro-sdis")
def save_split(dset, split):
    from PIL import Image
    for i, ex in enumerate(dset):
        img = ex["image"]; name = ex.get("image_name") or f"{split}_{i}.jpg"
        img.save(f"{OUT}/images/{split}/{name}")
        ann = ex.get("annotations","")
        with open(f"{OUT}/labels/{split}/{name.rsplit('.',1)[0]}.txt","w") as f:
            if isinstance(ann, str): f.write(ann+"\n")
            elif isinstance(ann, list): f.write("\n".join(map(str,ann))+"\n")
            else: pass
save_split(ds["train"], "train"); save_split(ds["val"], "val")
print("[✓] Pyro-SDIS exported to", OUT)
