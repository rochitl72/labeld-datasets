# labeld-datasets (unified)

Single unified dataset for **one class**: **physical median**.

## Structure (repo root)

- `images/{train,val,test}/` — images (numbered per-split)
- `labels/{train,val,test}/` — YOLO segmentation labels (`.txt`, class id `0`)
- `classes.txt` — class list (single class: `physical median`)
- `data.yaml` — YOLO dataset config
- `annotations_coco.json` — COCO-format annotations (single category: `physical median`)
- `manifest.json` — per-image metadata (includes `theme` to preserve the original source dataset)

## Splits

- **Total**: 501 images
- **Train**: 350 (70%)
- **Val**: 100 (20%)
- **Test**: 51 (10%)

## Notes

- **Shuffle seed**: 42 (deterministic split assignment)
- **File naming**: numbered **independently per split**, zero-padded for stable sorting: `0001.jpg`, `0002.jpg`, …
