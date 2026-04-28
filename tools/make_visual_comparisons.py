from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp")
MWIR = ROOT / "MWIR-Net"
PROMPTIR = ROOT / "PromptIR"
OUT_DIR = MWIR / "visual_comparisons"

CELL_W = 260
CELL_H = 170
HEADER_H = 42
LABEL_H = 24
PAD = 10
BG = (255, 255, 255)
TEXT = (31, 41, 55)
BORDER = (210, 218, 230)


def font(size=18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(17)
SMALL_FONT = font(14)


def fit_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_W - 2 * PAD, CELL_H - 2 * PAD), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - image.width) // 2
    y = (CELL_H - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_center(draw, xy, text, fnt, fill=TEXT):
    x, y, w, h = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=fill)


def derain_cases():
    ids = ["1.png", "10.png", "25.png", "50.png", "100.png"]
    columns = [
        ("Input", MWIR / "test/derain/Rain100L/input"),
        ("PromptIR Official", PROMPTIR / "official_output/derain"),
        ("PromptIR Init", PROMPTIR / "promptir_output_5k_12epoch_init/derain"),
        ("MWIR-Net Init", MWIR / "outputs/mwirnet_output_5k_12epoch_init/derain"),
        ("MWIR-Net TTA", MWIR / "outputs/mwirnet_output_stage2_charb_edge002_tta/derain"),
        ("GT", MWIR / "test/derain/Rain100L/target"),
    ]
    return "rain100l_comparison.png", ids, columns


def dehaze_cases():
    names = [
        "0001_0.8_0.2.png",
        "0030_0.95_0.12.png",
        "0066_1_0.08.png",
        "0100_0.9_0.12.png",
        "0056_0.8_0.16.png",
    ]
    columns = [
        ("Input", MWIR / "test/dehaze/outdoor/input"),
        ("PromptIR Official", PROMPTIR / "official_output/dehaze_outdoor"),
        ("PromptIR Init", PROMPTIR / "promptir_output_5k_12epoch_init/dehaze_outdoor"),
        ("MWIR-Net Init", MWIR / "outputs/mwirnet_output_5k_12epoch_init/dehaze_outdoor"),
        ("MWIR-Net TTA", MWIR / "outputs/mwirnet_output_stage2_charb_edge002_tta_dehaze/dehaze_outdoor"),
        ("GT", MWIR / "test/dehaze/outdoor/target"),
    ]
    return "sots_outdoor_comparison.png", names, columns


def resolve_path(name, title, directory):
    if title == "Input" and "dehaze" in str(directory):
        return directory / f"{Path(name).stem}.jpg"
    if title == "GT" and "dehaze" in str(directory):
        clean_id = Path(name).stem.split("_")[0]
        return directory / f"{clean_id}.png"
    return directory / name


def make_grid(output_name, names, columns):
    rows = len(names)
    cols = len(columns)
    width = cols * CELL_W
    height = HEADER_H + rows * (CELL_H + LABEL_H)
    grid = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(grid)

    for col_idx, (title, _) in enumerate(columns):
        x = col_idx * CELL_W
        draw.rectangle([x, 0, x + CELL_W - 1, HEADER_H - 1], fill=(234, 242, 255), outline=BORDER)
        draw_center(draw, (x, 0, CELL_W, HEADER_H), title, FONT)

    for row_idx, name in enumerate(names):
        y0 = HEADER_H + row_idx * (CELL_H + LABEL_H)
        for col_idx, (title, directory) in enumerate(columns):
            x0 = col_idx * CELL_W
            path = resolve_path(name, title, directory)
            if not path.exists():
                raise FileNotFoundError(path)
            image = fit_image(path)
            grid.paste(image, (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline=BORDER)
        label = Path(name).stem
        draw.rectangle([0, y0 + CELL_H, width - 1, y0 + CELL_H + LABEL_H - 1], fill=(246, 248, 251), outline=BORDER)
        draw_center(draw, (0, y0 + CELL_H, width, LABEL_H), label, SMALL_FONT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / output_name
    grid.save(out_path)
    print(out_path)


def main():
    for spec in [derain_cases(), dehaze_cases()]:
        make_grid(*spec)


if __name__ == "__main__":
    main()
