# labeld-datasets

Private backup of labeled datasets from **RBG Annotation Studio** (AnnoForge).

## Projects

| Folder | Description | Labeled images |
|--------|-------------|----------------|
| `LANE/` | LANE segmentation project | 329 |
| `white_separator/` | White separator project | 50 |
| `MEDIAN/` | MEDIAN segmentation project | 28 |
| `MEDIAN-1/` | MEDIAN-1 segmentation project | 94 |

Each project folder contains:

- `images/{train,val,test}/` — source images
- `labels/{train,val,test}/` — YOLO segmentation `.txt` files
- `annotations_coco.json` — COCO format annotations
- `classes.txt` — class names
- `data.yaml` — YOLO dataset config
- `manifest.json` — per-image metadata


Last updated: 2026-05-29
