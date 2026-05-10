#!/usr/bin/env python3
"""Create additional thesis-ready visual grids for MWIR-Net."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp/MWIR-Net")
OUT_DIR = ROOT / "visual_comparisons" / "extra_benchmarks"

CELL_W = 280
CELL_H = 180
HEADER_H = 42
LABEL_H = 24
PAD = 10

BG = (255, 255, 255)
HEADER_BG = (235, 242, 255)
LABEL_BG = (248, 250, 252)
BORDER = (203, 213, 225)
TEXT = (15, 23, 42)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = font(18, bold=True)
LABEL_FONT = font(14)
SMALL_FONT = font(13)


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    x, y, w, h = xy
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=TEXT)


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_W - 2 * PAD, CELL_H - 2 * PAD), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - image.width) // 2
    y = (CELL_H - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def sample_label(name: str) -> str:
    return Path(name).stem


def draw_headers(canvas: Image.Image, columns: list[str]) -> None:
    draw = ImageDraw.Draw(canvas)
    for col_idx, title in enumerate(columns):
        x = col_idx * CELL_W
        draw.rectangle([x, 0, x + CELL_W - 1, HEADER_H - 1], fill=HEADER_BG, outline=BORDER)
        draw_center(draw, (x, 0, CELL_W, HEADER_H), title, TITLE_FONT)


def make_grid(
    out_path: Path,
    title: str,
    samples: list[str],
    columns: list[str],
    resolve_path: Callable[[str, str], Path],
) -> None:
    width = len(columns) * CELL_W
    height = HEADER_H + len(samples) * (CELL_H + LABEL_H)
    canvas = Image.new("RGB", (width, height), BG)
    draw_headers(canvas, columns)
    draw = ImageDraw.Draw(canvas)

    for row_idx, sample in enumerate(samples):
        y0 = HEADER_H + row_idx * (CELL_H + LABEL_H)
        for col_idx, column in enumerate(columns):
            x0 = col_idx * CELL_W
            path = resolve_path(column, sample)
            if not path.exists():
                raise FileNotFoundError(path)
            canvas.paste(fit_image(path), (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline=BORDER)
        draw.rectangle([0, y0 + CELL_H, width - 1, y0 + CELL_H + LABEL_H - 1], fill=LABEL_BG, outline=BORDER)
        draw_center(draw, (0, y0 + CELL_H, width, LABEL_H), sample_label(sample), LABEL_FONT)

    banner = Image.new("RGB", (width, 32), BG)
    banner_draw = ImageDraw.Draw(banner)
    draw_center(banner_draw, (0, 0, width, 32), title, SMALL_FONT)

    full = Image.new("RGB", (width, height + 32), BG)
    full.paste(banner, (0, 0))
    full.paste(canvas, (0, 32))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.save(out_path)


def rain100l_ablation() -> None:
    samples = ["1.png", "25.png", "100.png"]
    columns = ["Input", "Zero Prompt", "No Channel Attention", "MWIR-Net", "Target"]

    def resolve(column: str, sample: str) -> Path:
        if column == "Input":
            return ROOT / "test/derain/Rain100L/input" / sample
        if column == "Zero Prompt":
            return ROOT / "outputs/mwirnet_output_ablation_zero_prompt/derain" / sample
        if column == "No Channel Attention":
            return ROOT / "outputs/mwirnet_output_ablation_no_channel_attention/derain" / sample
        if column == "MWIR-Net":
            return ROOT / "outputs/mwirnet_output_final_tta_multisplit/derain" / sample
        if column == "Target":
            return ROOT / "test/derain/Rain100L/target" / sample
        raise KeyError(column)

    make_grid(OUT_DIR / "rain100l_ablation_grid.png", "Rain100L ablation comparison", samples, columns, resolve)


def rain100h_tta() -> None:
    samples = ["1.png", "25.png", "100.png"]
    columns = ["Input", "MWIR-Net Plain", "MWIR-Net TTA", "Target"]

    def resolve(column: str, sample: str) -> Path:
        if column == "Input":
            return ROOT / "test/derain/Rain100H/input" / sample
        if column == "MWIR-Net Plain":
            return ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Rain100H" / sample
        if column == "MWIR-Net TTA":
            return ROOT / "outputs/mwirnet_output_final_tta_multisplit/derain_Rain100H" / sample
        if column == "Target":
            return ROOT / "test/derain/Rain100H/target" / sample
        raise KeyError(column)

    make_grid(OUT_DIR / "rain100h_plain_tta_grid.png", "Rain100H plain versus TTA", samples, columns, resolve)


def test1200_plain() -> None:
    samples = ["1.png", "600.png", "1200.png"]
    columns = ["Input", "MWIR-Net", "Target"]

    def resolve(column: str, sample: str) -> Path:
        if column == "Input":
            return ROOT / "test/derain/Test1200/input" / sample
        if column == "MWIR-Net":
            return ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Test1200" / sample
        if column == "Target":
            return ROOT / "test/derain/Test1200/target" / sample
        raise KeyError(column)

    make_grid(OUT_DIR / "test1200_grid.png", "Test1200 deraining comparison", samples, columns, resolve)


def test2800_plain() -> None:
    samples = ["801_1.jpg", "802_1.jpg", "803_1.jpg"]
    columns = ["Input", "MWIR-Net", "Target"]

    def resolve(column: str, sample: str) -> Path:
        stem = Path(sample).stem
        if column == "Input":
            return ROOT / "test/derain/Test2800/input" / sample
        if column == "MWIR-Net":
            return ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Test2800" / f"{stem}.png"
        if column == "Target":
            return ROOT / "test/derain/Test2800/target" / sample
        raise KeyError(column)

    make_grid(OUT_DIR / "test2800_grid.png", "Test2800 deraining comparison", samples, columns, resolve)


def nyuhaze500_plain() -> None:
    samples = ["1400_1.png", "1400_6.png", "1401_1.png"]
    columns = ["Input", "MWIR-Net", "Target"]

    def resolve(column: str, sample: str) -> Path:
        clean_id = Path(sample).stem.split("_")[0]
        if column == "Input":
            return ROOT / "test/dehaze/nyuhaze500/input" / sample
        if column == "MWIR-Net":
            return ROOT / "outputs/mwirnet_output_final_plain_multisplit_dehaze/dehaze_nyuhaze500" / sample
        if column == "Target":
            return ROOT / "test/dehaze/nyuhaze500/target" / f"{clean_id}.png"
        raise KeyError(column)

    make_grid(OUT_DIR / "nyuhaze500_grid.png", "NYU-Haze500 dehazing comparison", samples, columns, resolve)


def main() -> None:
    rain100l_ablation()
    rain100h_tta()
    test1200_plain()
    test2800_plain()
    nyuhaze500_plain()
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
