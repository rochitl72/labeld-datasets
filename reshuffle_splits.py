#!/usr/bin/env python3
"""
Reshuffle the unified dataset into a new 70/20/10 split and renumber filenames
independently within each split starting from 1.

This rewrites:
  - images/{train,val,test}/*
  - labels/{train,val,test}/*
  - manifest.json
  - annotations_coco.json
  - README.md (counts + note about shuffle seed)

Reproducibility: random seed = 42 by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Item:
    old_split: str
    old_stem: str
    img_path: Path
    label_path: Path
    theme: str | None


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


def list_items(repo: Path) -> list[Item]:
    # Load theme from manifest.json if present.
    theme_by_old: dict[tuple[str, str], str] = {}
    manifest_path = repo / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            if isinstance(manifest, list):
                for e in manifest:
                    # We key by old split + old filename stem when available (DJI_0001, etc.)
                    theme = e.get("theme")
                    if not theme:
                        continue
                    old_split = e.get("split")
                    old_manifest_filename = e.get("old_manifest_filename")
                    if old_split and old_manifest_filename:
                        theme_by_old[(str(old_split), Path(str(old_manifest_filename)).stem)] = str(theme)
        except Exception:
            pass

    items: list[Item] = []
    for split in SPLITS:
        img_dir = repo / "images" / split
        lbl_dir = repo / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        # Image stems can be any numeric file names already; labels are .txt.
        imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for img in imgs:
            stem = img.stem
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.exists():
                raise SystemExit(f"Missing label for image: {img} -> expected {lbl}")
            theme = theme_by_old.get((split, stem))
            items.append(Item(old_split=split, old_stem=stem, img_path=img, label_path=lbl, theme=theme))

    if not items:
        raise SystemExit(f"No dataset items found under {repo}/images/*")
    return items


def compute_split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Ratios must be in (0,1).")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0.")

    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    # Ensure we never produce empty splits unless n is tiny.
    return train_n, val_n, test_n


def rewrite_readme(repo: Path, train_n: int, val_n: int, test_n: int, seed: int) -> None:
    readme = repo / "README.md"
    if not readme.exists():
        return
    txt = readme.read_text(encoding="utf-8")

    # Replace any existing split line(s) in a robust way by appending/overwriting a section.
    # Keep it minimal: update the table counts if present, else append.
    total = train_n + val_n + test_n

    # If old table exists, rewrite it to a single project description.
    # Otherwise, append a short section.
    new_block = "\n".join(
        [
            "## Dataset",
            "",
            "Single unified dataset for one class: `physical median`.",
            "",
            f"- Total images: {total}",
            f"- Split: train {train_n} (70%), val {val_n} (20%), test {test_n} (10%)",
            f"- Shuffle seed: {seed}",
            "",
            "Notes:",
            "- Filenames are zero-padded and numbered independently within each split (`0001.jpg`, `0002.jpg`, ...).",
            "- `manifest.json` includes a `theme` field preserving the original source project.",
            "",
        ]
    )

    # Drop previous "Projects" section if present to avoid confusion.
    txt = re.sub(r"(?s)\n## Projects\n.*?(?=\n## |\Z)", "\n", txt)
    txt = txt.rstrip() + "\n\n" + new_block
    readme.write_text(txt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=".", help="Path to labeld-datasets repo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    items = list_items(repo)

    rng = random.Random(args.seed)
    rng.shuffle(items)

    train_n, val_n, test_n = compute_split_counts(len(items), args.train, args.val, args.test)
    train_items = items[:train_n]
    val_items = items[train_n : train_n + val_n]
    test_items = items[train_n + val_n :]

    # Prepare temp output
    tmp = repo / ".tmp_reshuffle"
    if tmp.exists():
        shutil.rmtree(tmp)
    for split in SPLITS:
        (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)

    new_manifest: list[dict[str, Any]] = []
    def emit(split: str, split_items: list[Item]) -> None:
        width = len(str(len(split_items)))
        for i, it in enumerate(split_items, start=1):
            stem = f"{i:0{width}d}"
            # Use consistent extension .jpg for outputs.
            new_img = tmp / "images" / split / f"{stem}.jpg"
            new_lbl = tmp / "labels" / split / f"{stem}.txt"

            shutil.copy2(it.img_path, new_img)
            remap_yolo_to_single_class(it.label_path, new_lbl)

            new_manifest.append(
                {
                    "split": split,
                    "split_id": i,
                    "file_name": f"{stem}.jpg",
                    "theme": it.theme,
                    "old_split": it.old_split,
                    "old_stem": it.old_stem,
                    "old_image_path": str(it.img_path.relative_to(repo)),
                    "old_label_path": str(it.label_path.relative_to(repo)),
                }
            )

    emit("train", train_items)
    emit("val", val_items)
    emit("test", test_items)

    # Rewrite COCO if present
    coco_path = repo / "annotations_coco.json"
    if coco_path.exists():
        coco = read_json(coco_path)
        old_images: list[dict[str, Any]] = coco.get("images", [])
        old_annotations: list[dict[str, Any]] = coco.get("annotations", [])

        # Build old (split, stem) -> new (split, file_name)
        old_to_new: dict[tuple[str, str], tuple[str, str]] = {}
        for e in new_manifest:
            old_to_new[(e["old_split"], e["old_stem"])] = (e["split"], str(e["file_name"]))

        # Old COCO file_name was "N.jpg" (no split); split is stored in image["split"].
        # Use image["split"] + file_name stem to map.
        new_images: list[dict[str, Any]] = []
        old_image_id_to_new_global: dict[int, int] = {}
        next_global_id = 1

        # Map every COCO image to a new one if it exists in the new manifest.
        for im in old_images:
            old_split = str(im.get("split") or "")
            old_stem = Path(str(im.get("file_name") or "")).stem
            mapped = old_to_new.get((old_split, old_stem))
            if not mapped:
                continue
            new_split, new_file_name = mapped
            old_image_id_to_new_global[int(im["id"])] = next_global_id
            im_out = dict(im)
            im_out["id"] = next_global_id
            im_out["split"] = new_split
            # Avoid COCO collisions across splits by including split in file_name.
            im_out["file_name"] = f"images/{new_split}/{new_file_name}"
            new_images.append(im_out)
            next_global_id += 1

        new_annotations: list[dict[str, Any]] = []
        next_ann_id = 1
        for ann in old_annotations:
            old_img_id = int(ann["image_id"])
            new_img_id = old_image_id_to_new_global.get(old_img_id)
            if new_img_id is None:
                continue
            ann_out = dict(ann)
            ann_out["id"] = next_ann_id
            next_ann_id += 1
            ann_out["image_id"] = new_img_id
            ann_out["category_id"] = 1
            new_annotations.append(ann_out)

        coco_out = {
            "info": coco.get("info") or {"description": "RBG physical median dataset", "version": "1.0"},
            "images": new_images,
            "annotations": new_annotations,
            "categories": [{"id": 1, "name": "physical median", "supercategory": "object"}],
        }
        write_json(coco_path, coco_out)

    # Replace dataset dirs atomically-ish: delete and move tmp into place
    for split in SPLITS:
        shutil.rmtree(repo / "images" / split)
        shutil.rmtree(repo / "labels" / split)
        (repo / "images" / split).parent.mkdir(parents=True, exist_ok=True)
        (repo / "labels" / split).parent.mkdir(parents=True, exist_ok=True)

    # Move temp dirs into place
    for split in SPLITS:
        shutil.move(str(tmp / "images" / split), str(repo / "images" / split))
        shutil.move(str(tmp / "labels" / split), str(repo / "labels" / split))

    shutil.rmtree(tmp)

    # Rewrite manifest.json
    write_json(repo / "manifest.json", new_manifest)

    # Ensure classes + data.yaml are correct
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

    rewrite_readme(repo, train_n, val_n, test_n, args.seed)

    print(f"Done. total={len(items)} train={train_n} val={val_n} test={test_n}")


if __name__ == "__main__":
    main()

