# Run Summary - exp_sanity10

This document explains each artifact and metric produced in `runs/detect/exp_sanity10` and summarizes your run's results.

## Overview

- Model: `yolov10n.pt`
- Dataset: `configs/dataset.resolved.yaml`
- Epochs: 5
- Image size: 512
- Batch size: 4
- Seed: 42 (deterministic)
- Augmentations: mosaic=0.8, mixup=0.1, fliplr=0.5

Training args saved in: `runs/detect/exp_sanity10/args.yaml`

## Key Metrics (last epoch)

Values taken from the last row of `results.csv`.

| Metric                       |      Value | What it means                                                                                         |
| ---------------------------- | ---------: | ----------------------------------------------------------------------------------------------------- |
| `metrics/precision(B)`       |    0.59961 | Fraction of predicted boxes that are correct (TP / (TP+FP)). Higher means fewer false positives.      |
| `metrics/recall(B)`          |    0.57610 | Fraction of ground-truth boxes detected (TP / (TP+FN)). Higher means fewer misses.                    |
| `metrics/mAP50(B)`           |    0.57208 | Mean Average Precision at IoU 0.50 across classes (PASCAL-style).                                     |
| `metrics/mAP50-95(B)`        |    0.29469 | Mean AP averaged over IoUs 0.50:0.95 (COCO-style). More stringent; sensitive to localization quality. |
| `train/box_loss`             |     3.6913 | Bounding-box regression loss. Lower indicates better localization learning.                           |
| `train/cls_loss`             |     3.1401 | Classification loss. Lower indicates better class probability calibration.                            |
| `train/dfl_loss`             |     2.0274 | Distribution Focal Loss for bounding-box edges (YOLOv8/10). Lower is better.                          |
| `val/box_loss`               |     3.2109 | Validation box loss (generalization of the regressor).                                                |
| `val/cls_loss`               |     3.4814 | Validation classification loss.                                                                       |
| `val/dfl_loss`               |     1.8200 | Validation DFL.                                                                                       |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | 0.00046676 | Learning rates for three parameter groups at the last epoch.                                          |

Notes:

- The "(B)" suffix denotes metrics computed using bounding boxes.
- mAP@0.5 is more forgiving; mAP@0.5:0.95 better reflects precise localization.

## Plots - what each shows

- `results.png`: Multi-panel summary of training/validation losses and metrics across epochs. Expect losses to trend down and precision/recall/mAP to trend up or stabilize.
- `PR_curve.png`: Precision-Recall curve over the validation set as confidence varies. Curves closer to the top-right indicate a better tradeoff. Area relates to AP.
- `P_curve.png`: Precision vs confidence threshold. Helps choose a threshold that limits false positives.
- `R_curve.png`: Recall vs confidence threshold. Shows how recall improves as threshold decreases.
- `F1_curve.png`: F1-score vs confidence threshold. The peak is a good default confidence threshold for deployment.

## Confusion matrices

- `confusion_matrix.png`: Raw counts of predictions-by-class vs ground-truth class at the evaluation IoU. Off-diagonal entries are confusions.
- `confusion_matrix_normalized.png`: Row-normalized percentages; easier to compare classes with different sample counts.

## Dataset diagnostics

- `labels.jpg`: Per-class instance counts and distributions of box sizes/aspects. Reveals class imbalance or prevalence of small objects.
- `labels_correlogram.jpg`: Class co-occurrence within images; highlights pairs of classes that frequently appear together.

## Batches (qualitative checks)

- `train_batch{0,1,2}.jpg`: Example training mosaics with augmentations. Verify box alignment and augmentation realism.
- `val_batch{n}_labels.jpg`: Ground-truth boxes on validation images (label audit).
- `val_batch{n}_pred.jpg`: Model predictions on the same images at default thresholds (qualitative performance).

## Weights and exports

- `weights/best.pt`: Best checkpoint during training (typically by mAP@0.5:0.95).
- `weights/last.pt`: Final checkpoint after the last epoch.
- `weights/best.onnx`: ONNX export of the best checkpoint for ONNX Runtime inference.

## results.csv - how to read

- Columns per epoch include: `epoch`, training losses (`train/*`), validation losses (`val/*`), metrics (`metrics/*(B)`), and learning rates (`lr/pg0..2`).
- Use the last row for "final" values, or take the maximum of metric columns for a "best epoch" snapshot.

## args.yaml - reproducibility snapshot

- Core: `model`, `data`, `epochs=5`, `batch=4`, `imgsz=512`, `workers=2`, `seed=42`.
- Optimization: `lr0=0.01`, `lrf=0.1`, `momentum=0.937`, `weight_decay=0.0005`.
- Augmentations: `mosaic=0.8`, `mixup=0.1`, `fliplr=0.5`.
- Logging/output: `project=runs/detect`, `name=exp_sanity10`, `plots=True`.

## Practical guidance

- If precision >> recall: model is conservative; lower confidence threshold or train longer to boost recall.
- If recall >> precision: many false positives; raise threshold, add hard negatives, refine labels.
- If mAP@0.5 is OK but mAP@0.5:0.95 is low: localization needs improvement; try more epochs, higher resolution, or stronger box-focused augmentation.
- If validation losses plateau while metrics rise: consider extending training or fine-tuning the LR schedule.

---

Generated for run: `runs/detect/exp_sanity10` using artifacts in the folder and the last row of `results.csv`.

---

python -c "import pandas as pd; df=pd.read_csv(r'runs\\detect\\exp_sanity10\\results.csv'); print(df.columns.tolist()); print(df.tail(1).to_dict('records')[0])"
[' epoch', ' train/box_loss', ' train/cls_loss', ' train/dfl_loss', ' metrics/precision(B)', ' metrics/recall(B)', ' metrics/mAP50(B)', ' metrics/mAP50-95(B)', ' val/box_loss', ' val/cls_loss', ' val/dfl_loss', ' lr/pg0', ' lr/pg1', ' lr/pg2']
{' epoch': 5, ' train/box_loss': 3.6913, ' train/cls_loss': 3.1401, ' train/dfl_loss': 2.0274, ' metrics/precision(B)': 0.59961, ' metrics/recall(B)': 0.5761, ' metrics/mAP50(B)': 0.57208, ' metrics/mAP50-95(B)': 0.29469, ' val/box_loss': 3.2109, ' val/cls_loss': 3.4814, ' val/dfl_loss': 1.82, ' lr/pg0': 0.00046676, ' lr/pg1': 0.00046676, ' lr/pg2': 0.00046676}

---

python src\run_eval_export.py --run runs\detect\exp_sanity10 --out out_sanity
[✓] Wrote out_sanity {'precision': 0.59961, 'recall': 0.5761, 'mAP50': 0.57208, 'mAP50-95': 0.29469, 'columns_seen': ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2']}
(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code> type .\out_sanity\metrics_summary.json

---

python src\update_word.py --doc "D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\paper\FireSmoke_Professional_Rich.docx" --metrics out_sanity\metrics_summary.json --pr out_sanity\pr_curve.png
[✓] Updated D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\paper\FireSmoke_Professional_Rich.docx
(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code> python src\update_latex.py --latex_dir "D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\paper\latex" --metrics out_sanity\metrics_summary.json
[✓] Wrote LaTeX results_table.tex
(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code>

---

python src\run_eval_export.py --run runs\detect\exp_sanity10 --out out_sanity

> > [✓] Wrote out_sanity {'precision': 0.59961, 'recall': 0.5761, 'mAP50': 0.57208, 'mAP50-95': 0.29469, 'columns_seen': ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2']}

## (venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code>

---

python src\train_yolov10.py --data configs\dataset.yaml --model yolov10n.pt `
--img 640 --epochs 50 --batch 8 --name exp_y10n_t1000 --workers 0
python src\run_eval_export.py --run runs\detect\exp_y10n_t1000 --out out
New https://pypi.org/project/ultralytics/8.3.195 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.0 🚀 Python-3.10.18 torch-2.3.1+cu121 CUDA:0 (Quadro T1000 with Max-Q Design, 4096MiB)
engine\trainer: task=detect, mode=train, model=yolov10n.pt, data=configs\dataset.resolved.yaml, epochs=50, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=0, project=runs/detect, name=exp_y10n_t1000, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=42, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=0.8, mixup=0.1, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\exp_y10n_t1000
Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]
1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]
2 -1 1 7360 ultralytics.nn.modules.block.C2f [32, 32, 1, True]
3 -1 1 18560 ultralytics.nn.modules.conv.Conv [32, 64, 3, 2]
4 -1 2 49664 ultralytics.nn.modules.block.C2f [64, 64, 2, True]
5 -1 1 9856 ultralytics.nn.modules.block.SCDown [64, 128, 3, 2]
6 -1 2 197632 ultralytics.nn.modules.block.C2f [128, 128, 2, True]
7 -1 1 36096 ultralytics.nn.modules.block.SCDown [128, 256, 3, 2]
8 -1 1 460288 ultralytics.nn.modules.block.C2f [256, 256, 1, True]
9 -1 1 164608 ultralytics.nn.modules.block.SPPF [256, 256, 5]
10 -1 1 249728 ultralytics.nn.modules.block.PSA [256, 256]
11 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
12 [-1, 6] 1 0 ultralytics.nn.modules.conv.Concat [1]
13 -1 1 148224 ultralytics.nn.modules.block.C2f [384, 128, 1]
14 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
15 [-1, 4] 1 0 ultralytics.nn.modules.conv.Concat [1]
16 -1 1 37248 ultralytics.nn.modules.block.C2f [192, 64, 1]
17 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]
18 [-1, 13] 1 0 ultralytics.nn.modules.conv.Concat [1]
19 -1 1 123648 ultralytics.nn.modules.block.C2f [192, 128, 1]
20 -1 1 18048 ultralytics.nn.modules.block.SCDown [128, 128, 3, 2]
21 [-1, 10] 1 0 ultralytics.nn.modules.conv.Concat [1]
22 -1 1 282624 ultralytics.nn.modules.block.C2fCIB [384, 256, 1, True, True]
23 [16, 19, 22] 1 862108 ultralytics.nn.modules.head.v10Detect [2, [64, 128, 256]]
YOLOv10n summary: 385 layers, 2,707,820 parameters, 2,707,804 gradients, 8.4 GFLOPs

Transferred 493/595 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code\data

---

python src\train_yolov10.py --data configs\dataset.yaml --model yolov10n.pt `

> > --img 640 --epochs 50 --batch 8 --name exp_y10n_t1000 --workers 0
> > python src\run_eval_export.py --run runs\detect\exp_y10n_t1000 --out out
> > New https://pypi.org/project/ultralytics/8.3.195 available 😃 Update with 'pip install -U ultralytics'
> > Ultralytics 8.3.0 🚀 Python-3.10.18 torch-2.3.1+cu121 CUDA:0 (Quadro T1000 with Max-Q Design, 4096MiB)
> > engine\trainer: task=detect, mode=train, model=yolov10n.pt, data=configs\dataset.resolved.yaml, epochs=50, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=0, project=runs/detect, name=exp_y10n_t1000, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=42, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=0.8, mixup=0.1, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\exp_y10n_t1000
> > Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]
1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]
2 -1 1 7360 ultralytics.nn.modules.block.C2f [32, 32, 1, True]
3 -1 1 18560 ultralytics.nn.modules.conv.Conv [32, 64, 3, 2]
4 -1 2 49664 ultralytics.nn.modules.block.C2f [64, 64, 2, True]
5 -1 1 9856 ultralytics.nn.modules.block.SCDown [64, 128, 3, 2]
6 -1 2 197632 ultralytics.nn.modules.block.C2f [128, 128, 2, True]
7 -1 1 36096 ultralytics.nn.modules.block.SCDown [128, 256, 3, 2]
8 -1 1 460288 ultralytics.nn.modules.block.C2f [256, 256, 1, True]
9 -1 1 164608 ultralytics.nn.modules.block.SPPF [256, 256, 5]
10 -1 1 249728 ultralytics.nn.modules.block.PSA [256, 256]
11 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
12 [-1, 6] 1 0 ultralytics.nn.modules.conv.Concat [1]
13 -1 1 148224 ultralytics.nn.modules.block.C2f [384, 128, 1]
14 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
15 [-1, 4] 1 0 ultralytics.nn.modules.conv.Concat [1]
16 -1 1 37248 ultralytics.nn.modules.block.C2f [192, 64, 1]
17 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]
18 [-1, 13] 1 0 ultralytics.nn.modules.conv.Concat [1]
19 -1 1 123648 ultralytics.nn.modules.block.C2f [192, 128, 1]
20 -1 1 18048 ultralytics.nn.modules.block.SCDown [128, 128, 3, 2]
21 [-1, 10] 1 0 ultralytics.nn.modules.conv.Concat [1]
22 -1 1 282624 ultralytics.nn.modules.block.C2fCIB [384, 256, 1, True, True]
23 [16, 19, 22] 1 862108 ultralytics.nn.modules.head.v10Detect [2, [64, 128, 256]]
YOLOv10n summary: 385 layers, 2,707,820 parameters, 2,707,804 gradients, 8.4 GFLOPs

Transferred 493/595 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code\data
train: New cache created: D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code\dataset\pyro_sdis_yolo\labels\train.cache
val: Scanning D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code\datase
val: New cache created: D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V3_1_Code\dataset\pyro_sdis_yolo\labels\val.cache
Plotting labels to runs\detect\exp_y10n_t1000\labels.jpg...
Plotting labels to runs\detect\exp_y10n_t1000\labels.jpg...
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 95 weight(decay=0.0), 108 weight(decay=0.0005), 107 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to runs\detect\exp_y10n_t1000
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50       1.6G        nan        nan        nan         11        640:  21%|██             1/50       1.6G        nan        nan        nan         11        640:  21%|██             1/50       1.6G        nan        nan        nan         11        640:  21%|██             1/50       1.6G        nan        nan        nan         15        640:  21%|██             1/50       1.6G        nan        nan        nan         15        640:  21%|██             1/50       1.6G        nan        nan        nan         11        640:  21%|██             1/50       1.6G        nan        nan        nan         11        640:  21%|██             1/50       1.6G        nan        nan        nan         14        640:  21%|██             1/50       1.6G        nan        nan        nan         14        640:  21%|██             1/50       1.6G        nan        nan        nan         14        640:  21%|██             1/50       1.6G        nan        nan        nan         14        640:  21%|██             1/50       1.6G        nan        nan        nan          9        640:  21%|██             1/50       1.6G        nan        nan        nan          9        640:  21%|██             1/50       1.6G        nan        nan        nan         15        640:  21%|██             1/50       1.6G        nan        nan        nan         15        640:  21%|██▏            1/50       1.6G        nan        nan        nan          9        640:  21%|██▏            1/50       1.6G        nan        nan        nan          9        640:  21%|██▏            1/50       1.6G        nan        nan        nan          8        640:  21%|██▏            1/50       1.6G        nan        nan        nan          8        640:  21%|██▏            1/50       1.6G        nan        nan        nan         10        640:  21%|██▏            1/50       1.6G        nan        nan        nan         10        640:  21%|██▏            1/50       1.6G        nan        nan        nan         10        640:  21%|██▏            1/50       1.6G        nan        nan        nan         10        640:  21%|██▏            1/50       1.6G        nan        nan        nan         12        640:  21%|██▏            1/50       1.6G        nan        nan        nan         12        640:  21%|██▏            1/50       1.6G        nan        nan        nan         11        640:  21%|██▏            1/50       1.6G        nan        nan        nan         11        640:  21%|██▏            1/50       1.6G        nan        nan        nan          9        640:  21%|██▏            1/50       1.6G        nan        nan        nan          9        640:  21%|██▏            1/50       1.6G        nan        nan        nan         11        640:  21%|██▏            1/50       1.6G        nan        nan        nan         11        640:  21%|██▏            1/50      1.61G        nan        nan        nan         19        640:  21%|██▏            1/50      1.61G        nan        nan        nan         19        640:  22%|██▏            1/50      1.61G        nan        nan        nan         15        640:  22%|██▏            1/50      1.61G        nan        nan        nan         15        640:  22%|██▏            1/50      1.61G        nan        nan        nan          8        640:  22%|██▏            1/50      1.61G        nan        nan        nan          8        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         10        640:  22%|██▏            1/50      1.61G        nan        nan        nan         10        640:  22%|██▏            1/50      1.61G        nan        nan        nan          8        640:  22%|██▏            1/50      1.61G        nan        nan        nan          8        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan         16        640:  22%|██▏            1/50      1.61G        nan        nan        nan         16        640:  22%|██▏            1/50      1.61G        nan        nan        nan         15        640:  22%|██▏            1/50      1.61G        nan        nan        nan         15        640:  22%|██▏            1/50      1.61G        nan        nan        nan          5        640:  22%|██▏            1/50      1.61G        nan        nan        nan          5        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan         11        640:  22%|██▏            1/50      1.61G        nan        nan        nan         11        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan         11        640:  22%|██▏            1/50      1.61G        nan        nan        nan         11        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         12        640:  22%|██▏            1/50      1.61G        nan        nan        nan         19        640:  22%|██▏            1/50      1.61G        nan        nan        nan         19        640:  22%|██▏            1/50      1.61G        nan        nan        nan         18        640:  22%|██▏            1/50      1.61G        nan        nan        nan         18        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         16        640:  22%|██▏            1/50      1.61G        nan        nan        nan         16        640:  22%|██▏            1/50      1.61G        nan        nan        nan         21        640:  22%|██▏            1/50      1.61G        nan        nan        nan         21        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan         13        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan         14        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan          9        640:  22%|██▏            1/50      1.61G        nan        nan        nan         10        640:  22%|██▏            1/50      1.61G        nan        nan        nan         10        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640:  22%|██▏            1/50      1.61G        nan        nan        nan         17        640       1/50      1.61G        nan        nan        nan         10        640:  33%|███▎      | 1224/3693 [18:59<35:48,
