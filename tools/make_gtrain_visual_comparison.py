#!/usr/bin/env python3
"""Create a GT-RAIN visual comparison grid for thesis figures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp/MWIR-Net")
INPUT_DIR = ROOT / "test/derain/GT-RAIN-test/input"
TARGET_DIR = ROOT / "test/derain/GT-RAIN-test/target"
OUTPUT_DIR = ROOT / "outputs/mwirnet_output_gtrain_plain/derain_GT-RAIN-test"
FIGURE_DIR = ROOT / "visual_comparisons"

CELL_W = 300
CELL_H = 220
HEADER_H = 44
LABEL_H = 28
PAD = 10
BG = (255, 255, 255)
TEXT = (31, 41, 55)
BORDER = (210, 218, 230)
HEADER_BG = (234, 242, 255)
LABEL_BG = (246, 248, 251)


def font(size: int = 18) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(17)
SMALL_FONT = font(13)


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str,
                fnt: ImageFont.ImageFont, fill: tuple[int, int, int] = TEXT) -> None:
    x, y, w, h = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=fill)


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_W - 2 * PAD, CELL_H - 2 * PAD), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - image.width) // 2
    y = (CELL_H - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def scene_name(path: Path) -> str:
    return path.name.split("__", 1)[0]


def frame_number(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("-R-", 1)[1])


def select_cases() -> list[Path]:
    by_scene: dict[str, list[Path]] = {}
    for output in sorted(OUTPUT_DIR.glob("*.png")):
        if (INPUT_DIR / output.name).exists() and (TARGET_DIR / output.name).exists():
            by_scene.setdefault(scene_name(output), []).append(output)

    if not by_scene:
        raise FileNotFoundError("No matched GT-RAIN input/output/target images found.")

    cases: list[Path] = []
    for _, outputs in sorted(by_scene.items()):
        outputs = sorted(outputs, key=frame_number)
        cases.append(outputs[len(outputs) // 2])
    return cases


def make_grid() -> Path:
    cases = select_cases()
    columns = [
        ("Rainy Input", INPUT_DIR),
        ("MWIR-Net", OUTPUT_DIR),
        ("Reference", TARGET_DIR),
    ]

    width = len(columns) * CELL_W
    height = HEADER_H + len(cases) * (CELL_H + LABEL_H)
    grid = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(grid)

    for col_idx, (title, _) in enumerate(columns):
        x = col_idx * CELL_W
        draw.rectangle([x, 0, x + CELL_W - 1, HEADER_H - 1], fill=HEADER_BG, outline=BORDER)
        draw_center(draw, (x, 0, CELL_W, HEADER_H), title, FONT)

    for row_idx, output in enumerate(cases):
        y0 = HEADER_H + row_idx * (CELL_H + LABEL_H)
        for col_idx, (_, directory) in enumerate(columns):
            x0 = col_idx * CELL_W
            path = directory / output.name
            if not path.exists():
                raise FileNotFoundError(path)
            grid.paste(fit_image(path), (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline=BORDER)

        label = f"{scene_name(output)}  frame R-{frame_number(output):03d}"
        draw.rectangle([0, y0 + CELL_H, width - 1, y0 + CELL_H + LABEL_H - 1],
                       fill=LABEL_BG, outline=BORDER)
        draw_center(draw, (0, y0 + CELL_H, width, LABEL_H), label, SMALL_FONT)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "gtrain_real_rain_comparison.png"
    grid.save(out_path)
    return out_path


def main() -> None:
    print(make_grid())


if __name__ == "__main__":
    main()
