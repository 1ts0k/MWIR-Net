#!/usr/bin/env python3
"""Create PPT-ready visual comparison strips for the 5k+5k protocol."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp")
MWIR = ROOT / "MWIR-Net"
PROMPTIR = ROOT / "PromptIR"
OUT_DIR = MWIR / "visual_comparisons" / "ppt_5k_protocol"

CELL_W = 320
CELL_H = 220
HEADER_H = 46
LABEL_H = 26
PAD = 10

BG = (255, 255, 255)
HEADER_BG = (239, 246, 255)
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


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    x, y, w, h = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=TEXT)


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_W - 2 * PAD, CELL_H - 2 * PAD), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - image.width) // 2
    y = (CELL_H - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def dehaze_input_name(name: str) -> str:
    return f"{Path(name).stem}.jpg"


def dehaze_clear_name(name: str) -> str:
    clean_id = Path(name).stem.split("_")[0]
    return f"{clean_id}.png"


def rain100l_spec() -> tuple[str, list[str], list[tuple[str, Path]]]:
    names = ["1.png", "10.png", "25.png", "50.png", "100.png"]
    columns = [
        ("Input", MWIR / "test/derain/Rain100L/input"),
        ("AirNet-5k", MWIR / "outputs/airnet_5k_12epoch/derain"),
        ("PromptIR-5k", PROMPTIR / "promptir_output_5k_12epoch_init/derain"),
        ("Ours", MWIR / "outputs/mwirnet_output_stage2_charb_edge002_tta/derain"),
        ("Clear", MWIR / "test/derain/Rain100L/target"),
    ]
    return "rain100l", names, columns


def sots_spec() -> tuple[str, list[str], list[tuple[str, Path]]]:
    names = [
        "0001_0.8_0.2.png",
        "0030_0.95_0.12.png",
        "0056_0.8_0.16.png",
        "0066_1_0.08.png",
        "0100_0.9_0.12.png",
    ]
    columns = [
        ("Input", MWIR / "test/dehaze/outdoor/input"),
        ("AirNet-5k", MWIR / "outputs/airnet_5k_12epoch/dehaze_outdoor"),
        ("PromptIR-5k", PROMPTIR / "promptir_output_5k_12epoch_init/dehaze_outdoor"),
        ("Ours", MWIR / "outputs/mwirnet_output_stage2_charb_edge002_tta_dehaze/dehaze_outdoor"),
        ("Clear", MWIR / "test/dehaze/outdoor/target"),
    ]
    return "sots_outdoor", names, columns


def resolve_path(dataset: str, title: str, directory: Path, name: str) -> Path:
    if dataset == "sots_outdoor" and title == "Input":
        return directory / dehaze_input_name(name)
    if dataset == "sots_outdoor" and title == "Clear":
        return directory / dehaze_clear_name(name)
    return directory / name


def row_label(dataset: str, name: str) -> str:
    if dataset == "rain100l":
        return f"Rain100L / {Path(name).stem}"
    if dataset == "sots_outdoor":
        return f"SOTS outdoor / {Path(name).stem}"
    return Path(name).stem


def draw_headers(canvas: Image.Image, columns: list[tuple[str, Path]]) -> None:
    draw = ImageDraw.Draw(canvas)
    for col_idx, (title, _) in enumerate(columns):
        x = col_idx * CELL_W
        draw.rectangle([x, 0, x + CELL_W - 1, HEADER_H - 1], fill=HEADER_BG, outline=BORDER)
        draw_center(draw, (x, 0, CELL_W, HEADER_H), title, TITLE_FONT)


def make_strip(dataset: str, name: str, columns: list[tuple[str, Path]]) -> Path:
    width = len(columns) * CELL_W
    height = HEADER_H + CELL_H + LABEL_H
    canvas = Image.new("RGB", (width, height), BG)
    draw_headers(canvas, columns)
    draw = ImageDraw.Draw(canvas)

    for col_idx, (title, directory) in enumerate(columns):
        x = col_idx * CELL_W
        path = resolve_path(dataset, title, directory, name)
        if not path.exists():
            raise FileNotFoundError(path)
        canvas.paste(fit_image(path), (x, HEADER_H))
        draw.rectangle([x, HEADER_H, x + CELL_W - 1, HEADER_H + CELL_H - 1], outline=BORDER)

    y = HEADER_H + CELL_H
    draw.rectangle([0, y, width - 1, y + LABEL_H - 1], fill=LABEL_BG, outline=BORDER)
    draw_center(draw, (0, y, width, LABEL_H), row_label(dataset, name), LABEL_FONT)

    output = OUT_DIR / f"{dataset}_{Path(name).stem}_strip.png"
    canvas.save(output)
    return output


def make_grid(dataset: str, names: list[str], columns: list[tuple[str, Path]]) -> Path:
    width = len(columns) * CELL_W
    height = HEADER_H + len(names) * (CELL_H + LABEL_H)
    canvas = Image.new("RGB", (width, height), BG)
    draw_headers(canvas, columns)
    draw = ImageDraw.Draw(canvas)

    for row_idx, name in enumerate(names):
        y0 = HEADER_H + row_idx * (CELL_H + LABEL_H)
        for col_idx, (title, directory) in enumerate(columns):
            x = col_idx * CELL_W
            path = resolve_path(dataset, title, directory, name)
            if not path.exists():
                raise FileNotFoundError(path)
            canvas.paste(fit_image(path), (x, y0))
            draw.rectangle([x, y0, x + CELL_W - 1, y0 + CELL_H - 1], outline=BORDER)

        y_label = y0 + CELL_H
        draw.rectangle([0, y_label, width - 1, y_label + LABEL_H - 1], fill=LABEL_BG, outline=BORDER)
        draw_center(draw, (0, y_label, width, LABEL_H), row_label(dataset, name), LABEL_FONT)

    output = OUT_DIR / f"{dataset}_ppt_grid.png"
    canvas.save(output)
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dataset, names, columns in [rain100l_spec(), sots_spec()]:
        written.append(make_grid(dataset, names, columns))
        for name in names:
            written.append(make_strip(dataset, name, columns))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
