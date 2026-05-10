#!/usr/bin/env python3
"""Copy a small curated set of raw samples into a tracked repository folder."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visual_comparisons" / "original_samples"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_group(
    group: str,
    input_dir: Path,
    target_dir: Path,
    samples: list[str],
    target_name: Callable[[str], str],
) -> list[Path]:
    written: list[Path] = []
    seen_targets: set[Path] = set()
    for sample in samples:
        copy_file(input_dir / sample, OUT / group / "input" / sample)
        written.append(OUT / group / "input" / sample)
        target_sample = target_name(sample)
        target_path = OUT / group / "target" / target_sample
        if target_path not in seen_targets:
            copy_file(target_dir / target_sample, target_path)
            written.append(target_path)
            seen_targets.add(target_path)
    return written


def main() -> None:
    jobs = [
        (
            "rain100l",
            ROOT / "test/derain/Rain100L/input",
            ROOT / "test/derain/Rain100L/target",
            ["1.png", "10.png", "25.png", "50.png", "100.png"],
            lambda sample: sample,
        ),
        (
            "rain100h",
            ROOT / "test/derain/Rain100H/input",
            ROOT / "test/derain/Rain100H/target",
            ["1.png", "25.png", "100.png"],
            lambda sample: sample,
        ),
        (
            "test1200",
            ROOT / "test/derain/Test1200/input",
            ROOT / "test/derain/Test1200/target",
            ["1.png", "600.png", "1200.png"],
            lambda sample: sample,
        ),
        (
            "test2800",
            ROOT / "test/derain/Test2800/input",
            ROOT / "test/derain/Test2800/target",
            ["801_1.jpg", "802_1.jpg", "803_1.jpg"],
            lambda sample: sample,
        ),
        (
            "sots_outdoor",
            ROOT / "test/dehaze/outdoor/input",
            ROOT / "test/dehaze/outdoor/target",
            [
                "0001_0.8_0.2.jpg",
                "0030_0.95_0.12.jpg",
                "0056_0.8_0.16.jpg",
                "0066_1_0.08.jpg",
                "0100_0.9_0.12.jpg",
            ],
            lambda sample: f"{Path(sample).stem.split('_')[0]}.png",
        ),
        (
            "nyuhaze500",
            ROOT / "test/dehaze/nyuhaze500/input",
            ROOT / "test/dehaze/nyuhaze500/target",
            ["1400_1.png", "1400_6.png", "1401_1.png"],
            lambda sample: f"{Path(sample).stem.split('_')[0]}.png",
        ),
    ]

    written: list[Path] = []
    for group, input_dir, target_dir, samples, target_name in jobs:
        written.extend(copy_group(group, input_dir, target_dir, samples, target_name))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
