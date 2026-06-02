#!/usr/bin/env python3
"""
Rebuild unified physical-median dataset from per-project exports.

Source format (per project):
  <project>/
    images/{train,val,test}/<coco_file_name>.jpg
    labels/{train,val,test}/<DJI_XXXX>.txt   (manifest filename stem)
    manifest.json                            (maps image_id -> filename + split)
    annotations_coco.json                    (maps image_id -> file_name + split)

Output (repo root):
  images/{train,val,test}/0001.jpg ...
  labels/{train,val,test}/0001.txt ...
  classes.txt (physical median)
  data.yaml
  manifest.json (includes theme)
  annotations_coco.json (file_name = images/<split>/<NNNN>.jpg; single category)

Split policy:
  - Pool all items across projects
  - Shuffle with seed (default 42)
  - Split by ratio (default 70/20/10 for train/val/test)
  - Renumber independently per split starting at 1 (zero-padded for stable sorting)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SrcItem:
    theme: str
    src_split: str
    src_project: Path
    src_img: Path
    src_lbl: Path
    old_image_id: int
    old_manifest_filename: str
    old_coco_file_name: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def remap_yolo_to_single_class(src: Path, dst: Path) -> None:
    out_lines: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            out_lines.append("0 " + " ".join(parts[1:]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def zero_pad(i: int, width: int) -> str:
    return str(i).zfill(width)


def gather_items(project_dir: Path) -> list[SrcItem]:
    theme = project_dir.name
    manifest = read_json(project_dir / "manifest.json")
    coco = read_json(project_dir / "annotations_coco.json")
    coco_images_by_id: dict[int, dict[str, Any]] = {int(im["id"]): im for im in coco["images"]}

    items: list[SrcItem] = []
    for entry in manifest:
        split = str(entry.get("split") or "")
        if split not in SPLITS:
            continue
        if int(entry.get("annotation_count", 0) or 0) <= 0:
            continue

        old_image_id = int(entry["image_id"])
        old_manifest_filename = str(entry.get("filename") or "")
        lbl_stem = Path(old_manifest_filename).stem  # e.g. DJI_0050
        coco_img = coco_images_by_id.get(old_image_id)
        if not coco_img:
            continue
        old_coco_file_name = str(coco_img["file_name"])  # actual exported image name

        img_path = project_dir / "images" / split / old_coco_file_name
        lbl_path = project_dir / "labels" / split / f"{lbl_stem}.txt"

        if not img_path.exists():
            raise SystemExit(f"Missing image file: {img_path}")
        if not lbl_path.exists():
            raise SystemExit(f"Missing label file: {lbl_path}")

        items.append(
            SrcItem(
                theme=theme,
                src_split=split,
                src_project=project_dir,
                src_img=img_path,
                src_lbl=lbl_path,
                old_image_id=old_image_id,
                old_manifest_filename=old_manifest_filename,
                old_coco_file_name=old_coco_file_name,
            )
        )
    return items


def compute_counts(n: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    train_n = int(n * train)
    val_n = int(n * val)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to repo working tree to rewrite (main)")
    ap.add_argument("--src", required=True, help="Path to old worktree containing per-project folders")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    src_root = Path(args.src).resolve()

    project_names = ["LANE", "MEDIAN", "MEDIAN-1", "1-June", "2-june"]
    all_items: list[SrcItem] = []
    for name in project_names:
        pd = src_root / name
        if not pd.exists():
            raise SystemExit(f"Missing project folder in src: {pd}")
        all_items.extend(gather_items(pd))

    if not all_items:
        raise SystemExit("No items gathered.")

    rng = random.Random(args.seed)
    rng.shuffle(all_items)

    train_n, val_n, test_n = compute_counts(len(all_items), args.train, args.val, args.test)
    splits: dict[str, list[SrcItem]] = {
        "train": all_items[:train_n],
        "val": all_items[train_n : train_n + val_n],
        "test": all_items[train_n + val_n :],
    }

    # Create tmp build dir
    tmp = repo / ".tmp_rebuild"
    if tmp.exists():
        shutil.rmtree(tmp)
    for s in SPLITS:
        (tmp / "images" / s).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Determine pad widths per split
    padw = {s: max(4, len(str(len(splits[s])))) for s in SPLITS}

    # Manifest & COCO rebuild
    out_manifest: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    next_coco_img_id = 1
    next_coco_ann_id = 1

    # Load per-project COCOs for annotations
    coco_by_project: dict[str, Any] = {name: read_json(src_root / name / "annotations_coco.json") for name in project_names}

    # Build index: project -> old_image_id -> image dict (width/height/status)
    coco_img_by_project_and_id: dict[tuple[str, int], dict[str, Any]] = {}
    for name in project_names:
        for im in coco_by_project[name]["images"]:
            coco_img_by_project_and_id[(name, int(im["id"]))] = im

    # Also need annotations grouped by (project, image_id)
    anns_by_project_and_image: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for name in project_names:
        for ann in coco_by_project[name]["annotations"]:
            key = (name, int(ann["image_id"]))
            anns_by_project_and_image.setdefault(key, []).append(ann)

    for split in SPLITS:
        for split_id, it in enumerate(splits[split], start=1):
            fname = f"{zero_pad(split_id, padw[split])}.jpg"
            out_img = tmp / "images" / split / fname
            out_lbl = tmp / "labels" / split / f"{zero_pad(split_id, padw[split])}.txt"

            shutil.copy2(it.src_img, out_img)
            remap_yolo_to_single_class(it.src_lbl, out_lbl)

            out_manifest.append(
                {
                    "split": split,
                    "split_id": split_id,
                    "file_name": fname,
                    "theme": it.theme,
                    "source_project": it.theme,
                    "old_image_id": it.old_image_id,
                    "old_manifest_filename": it.old_manifest_filename,
                    "old_coco_file_name": it.old_coco_file_name,
                }
            )

            src_im = coco_img_by_project_and_id[(it.theme, it.old_image_id)]
            coco_images.append(
                {
                    "id": next_coco_img_id,
                    "file_name": f"images/{split}/{fname}",
                    "width": src_im.get("width"),
                    "height": src_im.get("height"),
                    "split": split,
                    "status": src_im.get("status", "approved"),
                    "theme": it.theme,
                }
            )

            # Copy annotations, remap image_id + category_id
            for ann in anns_by_project_and_image.get((it.theme, it.old_image_id), []):
                ann_out = dict(ann)
                ann_out["id"] = next_coco_ann_id
                next_coco_ann_id += 1
                ann_out["image_id"] = next_coco_img_id
                ann_out["category_id"] = 1
                coco_annotations.append(ann_out)

            next_coco_img_id += 1

    # Replace repo dataset with tmp contents
    for s in SPLITS:
        if (repo / "images" / s).exists():
            shutil.rmtree(repo / "images" / s)
        if (repo / "labels" / s).exists():
            shutil.rmtree(repo / "labels" / s)
        (repo / "images" / s).parent.mkdir(parents=True, exist_ok=True)
        (repo / "labels" / s).parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(tmp / "images" / s), str(repo / "images" / s))
        shutil.move(str(tmp / "labels" / s), str(repo / "labels" / s))

    shutil.rmtree(tmp)

    # Write metadata
    (repo / "classes.txt").write_text("physical median\n", encoding="utf-8")
    (repo / "data.yaml").write_text(
        "\n".join(
            [
                "path: ./",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: ['physical median']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_json(repo / "manifest.json", out_manifest)
    write_json(
        repo / "annotations_coco.json",
        {
            "info": {
                "description": "Unified physical median dataset (rebuilt from per-project exports)",
                "version": "1.0",
                "seed": args.seed,
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "physical median", "supercategory": "object"}],
        },
    )

    total = len(all_items)
    print(f"rebuilt total={total} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()

