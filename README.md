# labeld-datasets (unified)

Single unified dataset for **one class**: **physical median**.

## Structure (repo root)

-  — images (numbered per-split)
-  — YOLO segmentation labels (, class id )
-  — class list (single class: )
-  — YOLO dataset config
-  — COCO-format annotations (single category: )
-  — per-image metadata (includes  to preserve the original source dataset)

## Splits

- **Total**: 637 images
- **Train**: 445 (70%)
- **Val**: 127 (20%)
- **Test**: 65 (10%)

## Notes

- **Shuffle seed**: 42 (deterministic split assignment)
- **File naming**: numbered **independently per split**, zero-padded for stable sorting: , , …
