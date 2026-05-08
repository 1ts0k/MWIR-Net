#!/usr/bin/env python3
"""Prepare local datasets for MWIR-Net.

The script creates symlinks instead of copying images, then rewrites the
list files under data_dir/. By default it reads datasets from
<project>/datasets:

  ITS_v2/{clear,hazy} or OTS_ALPHA/{clear,haze/OTS}
  SOTS/{nyuhaze500,outdoor}/{hazy,gt}
  RAIN13K/train/{input,target}
  RAIN13K/test/<split>/{input,target}
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def reset_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel_src)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_ots_train_list(project: Path) -> list[str]:
    list_file = project / "data_dir/hazy/hazy_outside.txt"
    if list_file.exists():
        return list_file.read_text(encoding="utf-8").splitlines()
    try:
        return subprocess.check_output(
            ["git", "show", "HEAD:data_dir/hazy/hazy_outside.txt"],
            cwd=project,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError("Missing OTS training list: data_dir/hazy/hazy_outside.txt") from exc


def prepare_derain(project: Path, workspace: Path) -> int:
    src_input = workspace / "RAIN13K/train/input"
    src_target = workspace / "RAIN13K/train/target"
    dst_rainy = project / "data/Train/Derain/rainy"
    dst_gt = project / "data/Train/Derain/gt"
    reset_dir(dst_rainy)
    reset_dir(dst_gt)

    lines: list[str] = []
    for degraded in image_files(src_input):
        clean = src_target / degraded.name
        if not clean.exists():
            continue
        rainy_name = f"rain-{degraded.stem}{degraded.suffix.lower()}"
        clean_name = f"norain-{degraded.stem}{degraded.suffix.lower()}"
        symlink(degraded, dst_rainy / rainy_name)
        symlink(clean, dst_gt / clean_name)
        lines.append(f"rainy/{rainy_name}")

    write_lines(project / "data_dir/rainy/rainTrain.txt", lines)
    return len(lines)


def prepare_dehaze_train(project: Path, workspace: Path, source: str) -> int:
    dst_original = project / "data/Train/Dehaze/original"
    dst_synthetic = project / "data/Train/Dehaze/synthetic"
    reset_dir(dst_original)
    reset_dir(dst_synthetic)

    if source == "its":
        src_clear = workspace / "ITS_v2/clear"
        src_hazy = workspace / "ITS_v2/hazy"
        clear_by_stem = {p.stem: p for p in image_files(src_clear)}
        lines: list[str] = []
        for hazy in image_files(src_hazy):
            clear_stem = hazy.stem.split("_")[0]
            if clear_stem not in clear_by_stem:
                continue
            symlink(hazy, dst_synthetic / hazy.name)
            lines.append(f"synthetic/{hazy.name}")
    elif source == "ots":
        src_clear_roots = [workspace / "OTS_ALPHA/clear", workspace / "OTS_ALPHA/clear/clear_images"]
        src_hazy = workspace / "OTS_ALPHA/haze/OTS"
        list_lines = read_ots_train_list(project)
        lines = [line.strip() for line in list_lines if line.strip()]
        required_ids = {Path(line).name.split("_")[0] for line in lines}
        clear_by_stem = {}
        for root in src_clear_roots:
            if root.exists():
                clear_by_stem.update({p.stem: p for p in image_files(root)})
        for line in lines:
            hazy = src_hazy / Path(line).name
            clear_stem = hazy.stem.split("_")[0]
            if clear_stem not in clear_by_stem:
                raise FileNotFoundError(f"Missing OTS clear image for id {clear_stem}")
            if not hazy.exists():
                raise FileNotFoundError(f"Missing OTS hazy image {hazy}")
            symlink(hazy, project / "data/Train/Dehaze" / line)
        clear_by_stem = {k: v for k, v in clear_by_stem.items() if k in required_ids}
    else:
        raise ValueError("--dehaze-source must be 'its' or 'ots'")

    for clear in clear_by_stem.values():
        symlink(clear, dst_original / clear.name)

    write_lines(project / "data_dir/hazy/hazy_outside.txt", lines)
    return len(lines)


def prepare_derain_tests(project: Path, workspace: Path) -> dict[str, int]:
    test_root = workspace / "RAIN13K/test"
    dst_root = project / "test/derain"
    counts: dict[str, int] = {}
    for split in sorted(p for p in test_root.iterdir() if p.is_dir()):
        dst_input = dst_root / split.name / "input"
        dst_target = dst_root / split.name / "target"
        reset_dir(dst_input)
        reset_dir(dst_target)
        count = 0
        for degraded in image_files(split / "input"):
            clean = split / "target" / degraded.name
            if not clean.exists():
                continue
            symlink(degraded, dst_input / degraded.name)
            symlink(clean, dst_target / degraded.name)
            count += 1
        counts[split.name] = count
    return counts


def prepare_dehaze_tests(project: Path, workspace: Path) -> dict[str, int]:
    dst_root = project / "test/dehaze"
    counts: dict[str, int] = {}
    for split_name in ["nyuhaze500", "outdoor"]:
        src_split = workspace / "SOTS" / split_name
        if not src_split.exists():
            continue
        dst_input = dst_root / split_name / "input"
        dst_target = dst_root / split_name / "target"
        reset_dir(dst_input)
        reset_dir(dst_target)

        gt_by_stem = {p.stem: p for p in image_files(src_split / "gt")}
        for gt in gt_by_stem.values():
            symlink(gt, dst_target / gt.name)

        count = 0
        for hazy in image_files(src_split / "hazy"):
            clean_stem = hazy.stem.split("_")[0]
            if clean_stem not in gt_by_stem:
                continue
            symlink(hazy, dst_input / hazy.name)
            count += 1
        counts[split_name] = count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dehaze-source", choices=["its", "ots"], default="its",
                        help="dehaze training source. Use ots for RESIDE OTS training.")
    args = parser.parse_args()

    project = args.project_root.resolve()
    workspace = args.workspace_root.resolve()

    derain_train = prepare_derain(project, workspace)
    dehaze_train = prepare_dehaze_train(project, workspace, args.dehaze_source)
    derain_tests = prepare_derain_tests(project, workspace)
    dehaze_tests = prepare_dehaze_tests(project, workspace)

    print(f"Derain train pairs: {derain_train}")
    print(f"Dehaze train pairs: {dehaze_train}")
    print("Derain test pairs:", derain_tests)
    print("Dehaze test hazy inputs:", dehaze_tests)


if __name__ == "__main__":
    main()
