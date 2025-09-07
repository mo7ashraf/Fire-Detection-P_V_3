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

| Metric | Value | What it means |
|---|---:|---|
| `metrics/precision(B)` | 0.59961 | Fraction of predicted boxes that are correct (TP / (TP+FP)). Higher means fewer false positives. |
| `metrics/recall(B)` | 0.57610 | Fraction of ground-truth boxes detected (TP / (TP+FN)). Higher means fewer misses. |
| `metrics/mAP50(B)` | 0.57208 | Mean Average Precision at IoU 0.50 across classes (PASCAL-style). |
| `metrics/mAP50-95(B)` | 0.29469 | Mean AP averaged over IoUs 0.50:0.95 (COCO-style). More stringent; sensitive to localization quality. |
| `train/box_loss` | 3.6913 | Bounding-box regression loss. Lower indicates better localization learning. |
| `train/cls_loss` | 3.1401 | Classification loss. Lower indicates better class probability calibration. |
| `train/dfl_loss` | 2.0274 | Distribution Focal Loss for bounding-box edges (YOLOv8/10). Lower is better. |
| `val/box_loss` | 3.2109 | Validation box loss (generalization of the regressor). |
| `val/cls_loss` | 3.4814 | Validation classification loss. |
| `val/dfl_loss` | 1.8200 | Validation DFL. |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | 0.00046676 | Learning rates for three parameter groups at the last epoch. |

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

