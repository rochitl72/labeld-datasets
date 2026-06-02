# Unified Labeled Dataset (Physical Median)

This repository contains a **single unified dataset** for **physical median** detection/segmentation.

## Contents (repo root)

- `images/` — images
- `labels/` — labels/annotations (as exported)
- `annotations_coco.json` — COCO-format annotations
- `manifest.json` — dataset manifest; includes a `theme` field to preserve the original source project/theme
- `classes.txt` — class list (single class: physical median)
- `data.yaml` — dataset config

## Dataset split counts

- **Total**: 501 images
- **Train**: 456 images
- **Val**: 30 images
- **Test**: 15 images

## Notes

- Filenames are normalized to **numeric** identifiers for consistency.
