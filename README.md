# labeld-datasets

Private backup of labeled datasets from **RBG Annotation Studio** (AnnoForge).

## Projects

| Folder | Description | Labeled images |
|--------|-------------|----------------|
| `LANE/` | LANE segmentation project | 329 |
| `white_separator/` | White separator project | 50 |

Each project folder contains:

- `images/{train,val,test}/` — source images
- `labels/{train,val,test}/` — YOLO segmentation `.txt` files
- `annotations_coco.json` — COCO format annotations
- `classes.txt` — class names
- `data.yaml` — YOLO dataset config
- `manifest.json` — per-image metadata

## Database snapshot

`metadata/annoforge.db` — full SQLite database (all projects, annotations, statuses) at time of backup.

## Restore locally

Copy `metadata/annoforge.db` to `annoforge/backend/annoforge.db` and image files from `storage/project_*` are referenced by paths inside the DB; for training, use the exported `images/` + `labels/` folders directly.

Backup created: 2026-05-26
