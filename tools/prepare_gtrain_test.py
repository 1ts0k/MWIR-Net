#!/usr/bin/env python3
"""Prepare GT-RAIN scenes as a MWIR-Net deraining test split.

GT-RAIN provides one clean reference frame and many rainy frames per scene.
This script creates the input/target layout expected by test.py:

  test/derain/GT-RAIN-test/input/<scene>__<rain-frame>.png
  test/derain/GT-RAIN-test/target/<scene>__<rain-frame>.png

The target image is the scene clean frame reused for each rainy frame. The
dataset is useful for real-rain visual analysis; objective metrics should be
reported cautiously because frame alignment can vary across scenes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_DIRS = {
    "train": "GT-RAIN_train",
    "val": "GT-RAIN_val",
    "test": "GT-RAIN_test",
}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel_src)


def prepare_scene(scene: Path, output_root: Path, max_per_scene: int) -> int:
    clear_frames = [p for p in image_files(scene) if "-Webcam-C-" in p.name]
    rainy_frames = [p for p in image_files(scene) if "-Webcam-R-" in p.name]
    if not clear_frames or not rainy_frames:
        return 0

    clean = clear_frames[0]
    if max_per_scene > 0:
        rainy_frames = rainy_frames[:max_per_scene]

    count = 0
    for rainy in rainy_frames:
        name = f"{scene.name}__{rainy.name}"
        symlink(rainy, output_root / "input" / name)
        symlink(clean, output_root / "target" / name)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GT-RAIN for MWIR-Net testing.")
    parser.add_argument("--root", type=Path, default=Path("datasets/GT-RAIN"))
    parser.add_argument("--split", choices=sorted(SPLIT_DIRS), default="test")
    parser.add_argument("--output-root", type=Path, default=Path("test/derain"))
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--max-per-scene", type=int, default=0,
                        help="0 means use all rainy frames in each scene.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.root / SPLIT_DIRS[args.split]
    output_name = args.output_name or f"GT-RAIN-{args.split}"
    output_root = args.output_root / output_name

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing GT-RAIN split directory: {split_dir}")

    total = 0
    scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for scene in scenes:
        total += prepare_scene(scene, output_root, args.max_per_scene)

    print(f"Prepared {total} rainy frames from {len(scenes)} scenes under {output_root}")
    print("Use with: python test.py --mode 1 --derain_splits", output_name)


if __name__ == "__main__":
    main()
