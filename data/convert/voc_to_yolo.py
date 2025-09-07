#!/usr/bin/env python3
import os, argparse, xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
def convert_bbox(size, box):
    W,H = size; dw, dh = 1.0/W, 1.0/H
    x=(box[0]+box[2])/2.0; y=(box[1]+box[3])/2.0
    w=box[2]-box[0]; h=box[3]-box[1]
    return (x*dw,y*dh,w*dw,h*dh)
def main(voc_dir, out_dir):
    (Path(out_dir)/"images/train").mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/"images/val").mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/"labels/train").mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/"labels/val").mkdir(parents=True, exist_ok=True)
    def split_of(name): return "val" if (hash(name)%100)<15 else "train"
    xmls = list(Path(voc_dir).rglob("*.xml"))
    for xml in xmls:
        root = ET.parse(xml).getroot()
        filename = root.findtext("filename")
        img_path = xml.parent/filename
        if not img_path.exists():
            cand = list(xml.parent.glob(Path(filename).stem+".*"))
            if not cand: continue
            img_path=cand[0]
        img = Image.open(img_path); W,H = img.size
        split = split_of(filename); img.save(Path(out_dir)/f"images/{split}/{img_path.name}")
        lines=[]
        for obj in root.findall("object"):
            name = obj.findtext("name").lower().strip()
            cls = 0 if name in ["smoke","wildfire_smoke","smoke_plume"] else 1
            b = obj.find("bndbox"); xmin=float(b.findtext("xmin")); ymin=float(b.findtext("ymin")); xmax=float(b.findtext("xmax")); ymax=float(b.findtext("ymax"))
            x,y,w,h = convert_bbox((W,H),(xmin,ymin,xmax,ymax))
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        with open(Path(out_dir)/f"labels/{split}/{img_path.stem}.txt","w") as f:
            f.write("\n".join(lines))
    print("[✓] VOC→YOLO at", out_dir)
if __name__=="__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--voc_dir", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); main(args.voc_dir, args.out)
