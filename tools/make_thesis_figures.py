#!/usr/bin/env python3
"""Generate thesis figures for MWIR-Net documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def add_box(ax, xy, width, height, label, facecolor, edgecolor="#1f2933"):
    patch = Rectangle(xy, width, height, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="#102a43",
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#334e68")
    ax.add_patch(arrow)


def make_architecture(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), dpi=180)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y = 2.0
    add_box(ax, (0.3, y), 1.1, 0.8, "Input\nFog/Rain", "#d9e2ec")
    add_box(ax, (1.8, y), 1.4, 0.8, "Overlap\nPatch Embed", "#bcccdc")
    add_box(ax, (3.6, 3.1), 1.3, 0.75, "Encoder L1\nTransformer", "#f0f4f8")
    add_box(ax, (5.2, 3.1), 1.3, 0.75, "Encoder L2\nTransformer", "#f0f4f8")
    add_box(ax, (6.8, 3.1), 1.3, 0.75, "Encoder L3\nTransformer", "#f0f4f8")
    add_box(ax, (8.4, 3.1), 1.3, 0.75, "Latent\nTransformer", "#f0f4f8")

    add_box(ax, (8.4, 1.55), 1.3, 0.75, "Weather\nPrompt L3", "#d9f99d")
    add_box(ax, (6.8, 1.55), 1.3, 0.75, "Weather\nPrompt L2", "#d9f99d")
    add_box(ax, (5.2, 1.55), 1.3, 0.75, "Weather\nPrompt L1", "#d9f99d")
    add_box(ax, (6.0, 0.45), 2.2, 0.65, "Prompt Channel Attention", "#bbf7d0")

    add_box(ax, (8.4, y), 1.3, 0.8, "Decoder L3\nFusion", "#fde68a")
    add_box(ax, (6.8, y), 1.3, 0.8, "Decoder L2\nFusion", "#fde68a")
    add_box(ax, (5.2, y), 1.3, 0.8, "Decoder L1\nFusion", "#fde68a")
    add_box(ax, (10.2, y), 1.2, 0.8, "Refinement\nBlocks", "#fed7aa")
    add_box(ax, (11.8, y), 1.0, 0.8, "Output\nRestored", "#bae6fd")

    add_arrow(ax, (1.4, y + 0.4), (1.8, y + 0.4))
    add_arrow(ax, (3.2, y + 0.4), (3.6, 3.48))
    add_arrow(ax, (4.9, 3.48), (5.2, 3.48))
    add_arrow(ax, (6.5, 3.48), (6.8, 3.48))
    add_arrow(ax, (8.1, 3.48), (8.4, 3.48))
    add_arrow(ax, (9.05, 3.1), (9.05, 2.8))
    add_arrow(ax, (9.05, 2.35), (9.05, 2.8))
    add_arrow(ax, (9.7, y + 0.4), (10.2, y + 0.4))
    add_arrow(ax, (11.4, y + 0.4), (11.8, y + 0.4))
    add_arrow(ax, (9.05, 1.55), (8.2, 1.05))
    add_arrow(ax, (7.45, 1.55), (7.45, 1.1))
    add_arrow(ax, (5.85, 1.55), (6.0, 1.1))

    ax.text(0.3, 4.45, "MWIR-Net: Multi-scale Weather-aware Image Restoration Network", fontsize=13, weight="bold")
    ax.text(0.3, 0.25, "Training: Charbonnier reconstruction loss + Sobel edge consistency. Inference: optional 8-way TTA self-ensemble.", fontsize=9, color="#52616b")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), label, fill=(20, 30, 40))
    return canvas


def make_gtrain_examples(gtrain_root: Path, output_path: Path) -> None:
    split_root = gtrain_root / "GT-RAIN_test"
    scene = next((p for p in sorted(split_root.iterdir()) if p.is_dir()), None)
    if scene is None:
        raise FileNotFoundError(f"No scene directories found in {split_root}")

    clean = next((p for p in image_files(scene) if "-Webcam-C-" in p.name), None)
    rainy = [p for p in image_files(scene) if "-Webcam-R-" in p.name][:2]
    if clean is None or len(rainy) < 2:
        raise FileNotFoundError(f"Scene {scene} does not contain one clean and two rainy frames")

    images = [
        draw_label(Image.open(clean).convert("RGB").resize((300, 190)), "Reference frame"),
        draw_label(Image.open(rainy[0]).convert("RGB").resize((300, 190)), "Real rain frame 1"),
        draw_label(Image.open(rainy[1]).convert("RGB").resize((300, 190)), "Real rain frame 2"),
    ]

    gap = 12
    canvas = Image.new("RGB", (sum(img.width for img in images) + gap * 2, images[0].height), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MWIR-Net thesis figures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--gtrain-root", type=Path, default=ROOT / "datasets/GT-RAIN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_architecture(args.output_dir / "mwirnet_architecture.png")
    make_gtrain_examples(args.gtrain_root, args.output_dir / "gtrain_real_rain_examples.png")
    print(f"Wrote thesis figures to {args.output_dir}")


if __name__ == "__main__":
    main()
