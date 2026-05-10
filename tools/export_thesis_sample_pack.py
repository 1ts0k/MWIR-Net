#!/usr/bin/env python3
"""Export a thesis sample pack with original pairs and model outputs.

The package is intentionally small enough to keep in GitHub while still
containing the raw pairs and the model results that are referenced in the
thesis figures.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PROMPTIR = ROOT.parent / "PromptIR"
OUT_ROOT = ROOT / "visual_comparisons" / "thesis_sample_pack"

CELL_W = 240
CELL_H = 180
HEADER_H = 40
LABEL_H = 24
TITLE_H = 32
PAD = 10

BG = (255, 255, 255)
HEADER_BG = (235, 242, 255)
LABEL_BG = (248, 250, 252)
BORDER = (203, 213, 225)
TEXT = (15, 23, 42)


@dataclass(frozen=True)
class SampleSpec:
    input_name: str
    target_name: str
    output_name: str

    @property
    def label(self) -> str:
        return Path(self.input_name).stem


@dataclass(frozen=True)
class ColumnSpec:
    label: str
    source_dir: Path
    file_name: Callable[[SampleSpec], str]
    export_folder: str
    asset_kind: str


@dataclass(frozen=True)
class GroupSpec:
    name: str
    title: str
    input_dir: Path
    target_dir: Path
    samples: list[SampleSpec]
    output_columns: list[ColumnSpec]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = font(18, bold=True)
HEADER_FONT = font(14, bold=True)
LABEL_FONT = font(13)


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_W - 2 * PAD, CELL_H - 2 * PAD), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - image.width) // 2
    y = (CELL_H - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    x, y, w, h = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=TEXT)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_output_name(sample: SampleSpec) -> str:
    return sample.output_name


def input_asset(group: GroupSpec) -> ColumnSpec:
    return ColumnSpec(
        label="Input",
        source_dir=group.input_dir,
        file_name=lambda sample: sample.input_name,
        export_folder="input",
        asset_kind="input",
    )


def target_asset(group: GroupSpec) -> ColumnSpec:
    return ColumnSpec(
        label="Target",
        source_dir=group.target_dir,
        file_name=lambda sample: sample.target_name,
        export_folder="target",
        asset_kind="target",
    )


def unique_columns(group: GroupSpec) -> list[ColumnSpec]:
    return [input_asset(group), *group.output_columns, target_asset(group)]


def resolve_source_path(column: ColumnSpec, sample: SampleSpec) -> Path:
    return column.source_dir / column.file_name(sample)


def export_group(group: GroupSpec) -> list[dict[str, str]]:
    group_root = OUT_ROOT / group.name
    rows: list[dict[str, str]] = []
    seen_targets: set[str] = set()

    # Copy input, outputs and targets.
    for sample in group.samples:
        for column in unique_columns(group):
            source_path = resolve_source_path(column, sample)
            if column.asset_kind == "output":
                export_path = group_root / "outputs" / column.export_folder / column.file_name(sample)
            else:
                export_path = group_root / column.export_folder / column.file_name(sample)
            if column.asset_kind == "target":
                key = export_path.name
                if key in seen_targets:
                    continue
                seen_targets.add(key)
            copy_file(source_path, export_path)
            rows.append(
                {
                    "group": group.name,
                    "sample": sample.label,
                    "asset_type": column.asset_kind,
                    "column": column.label,
                    "source_path": str(source_path.relative_to(ROOT.parent)),
                    "export_path": str(export_path.relative_to(ROOT.parent)),
                    "preview_path": str((group_root / "preview" / f"{sample.label}.png").relative_to(ROOT.parent)),
                    "note": "",
                }
            )

    # Build a preview collage for each sample.
    for sample in group.samples:
        preview_path = group_root / "preview" / f"{sample.label}.png"
        preview_columns = unique_columns(group)
        width = len(preview_columns) * CELL_W
        height = TITLE_H + HEADER_H + CELL_H + LABEL_H
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, width - 1, TITLE_H - 1], fill=BG)
        draw_center(draw, (0, 0, width, TITLE_H), group.title, TITLE_FONT)

        for col_idx, column in enumerate(preview_columns):
            x = col_idx * CELL_W
            draw.rectangle([x, TITLE_H, x + CELL_W - 1, TITLE_H + HEADER_H - 1], fill=HEADER_BG, outline=BORDER)
            draw_center(draw, (x, TITLE_H, CELL_W, HEADER_H), column.label, HEADER_FONT)

            source_path = resolve_source_path(column, sample)
            image = fit_image(source_path)
            y0 = TITLE_H + HEADER_H
            canvas.paste(image, (x, y0))
            draw.rectangle([x, y0, x + CELL_W - 1, y0 + CELL_H - 1], outline=BORDER)

        y_label = TITLE_H + HEADER_H + CELL_H
        draw.rectangle([0, y_label, width - 1, y_label + LABEL_H - 1], fill=LABEL_BG, outline=BORDER)
        draw_center(draw, (0, y_label, width, LABEL_H), sample.label, LABEL_FONT)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(preview_path)
        rows.append(
            {
                "group": group.name,
                "sample": sample.label,
                "asset_type": "preview",
                "column": "preview",
                "source_path": "",
                "export_path": str(preview_path.relative_to(ROOT.parent)),
                "preview_path": str(preview_path.relative_to(ROOT.parent)),
                "note": "",
            }
        )

    return rows


def build_readme(groups: list[GroupSpec]) -> str:
    lines = [
        "# Thesis Sample Pack",
        "",
        "This folder keeps a compact set of paper-ready original pairs and model outputs.",
        "It mirrors the figures referenced in the thesis without copying the full output archive.",
        "Each group follows `input/`, `target/`, `outputs/<model>/`, and `preview/`.",
        "Preview collages use the same left-to-right column order listed in the table below.",
        "",
        "| Group | Samples | Columns |",
        "|---|---|---|",
    ]
    for group in groups:
        column_labels = ", ".join(column.label for column in unique_columns(group))
        sample_labels = ", ".join(sample.label for sample in group.samples)
        lines.append(f"| `{group.name}` | `{sample_labels}` | {column_labels} |")

    lines += [
        "",
        "Regenerate the pack with:",
        "",
        "```bash",
        "python tools/export_thesis_sample_pack.py",
        "```",
        "",
        "The `manifest.csv` file records every copied image and preview path.",
    ]
    return "\n".join(lines) + "\n"


def build_manifest_csv(rows: list[dict[str, str]]) -> str:
    fieldnames = ["group", "sample", "asset_type", "column", "source_path", "export_path", "preview_path", "note"]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def make_sample(input_name: str, target_name: str | None = None, output_name: str | None = None) -> SampleSpec:
    target_name = target_name or input_name
    output_name = output_name or Path(input_name).with_suffix(".png").name
    return SampleSpec(input_name=input_name, target_name=target_name, output_name=output_name)


def group_specs() -> list[GroupSpec]:
    rain100l_samples = [make_sample(f"{i}.png") for i in [1, 10, 25, 50, 100]]
    sots_samples = [
        make_sample("0001_0.8_0.2.jpg", "0001.png", "0001_0.8_0.2.png"),
        make_sample("0030_0.95_0.12.jpg", "0030.png", "0030_0.95_0.12.png"),
        make_sample("0056_0.8_0.16.jpg", "0056.png", "0056_0.8_0.16.png"),
        make_sample("0066_1_0.08.jpg", "0066.png", "0066_1_0.08.png"),
        make_sample("0100_0.9_0.12.jpg", "0100.png", "0100_0.9_0.12.png"),
    ]
    rain100h_samples = [make_sample(f"{i}.png") for i in [1, 25, 100]]
    test1200_samples = [make_sample(f"{i}.png") for i in [1, 600, 1200]]
    test2800_samples = [
        make_sample("801_1.jpg", "801_1.jpg", "801_1.png"),
        make_sample("802_1.jpg", "802_1.jpg", "802_1.png"),
        make_sample("803_1.jpg", "803_1.jpg", "803_1.png"),
    ]
    nyu_samples = [
        make_sample("1400_1.png", "1400.png", "1400_1.png"),
        make_sample("1400_6.png", "1400.png", "1400_6.png"),
        make_sample("1401_1.png", "1401.png", "1401_1.png"),
    ]

    rain100l_main = GroupSpec(
        name="rain100l_main",
        title="Rain100L main comparison",
        input_dir=ROOT / "test/derain/Rain100L/input",
        target_dir=ROOT / "test/derain/Rain100L/target",
        samples=rain100l_samples,
        output_columns=[
            ColumnSpec("Restormer-official", ROOT / "outputs/restormer_official_derain_rain100l/Deraining", default_output_name, "restormer_official", "output"),
            ColumnSpec("PromptIR-official", PROMPTIR / "official_output/derain", default_output_name, "promptir_official", "output"),
            ColumnSpec("MPRNet-official", ROOT / "outputs/mprnet_official_derain_rain100l", default_output_name, "mprnet_official", "output"),
            ColumnSpec("AirNet-official-All", ROOT / "outputs/airnet_official_all/derain", default_output_name, "airnet_official_all", "output"),
            ColumnSpec("PromptIR-5k-init", PROMPTIR / "promptir_output_5k_12epoch_init/derain", default_output_name, "promptir_5k_init", "output"),
            ColumnSpec("AirNet-5k", ROOT / "outputs/airnet_5k_12epoch/derain", default_output_name, "airnet_5k", "output"),
            ColumnSpec("MWIR-Net-5k-init", ROOT / "outputs/mwirnet_output_5k_12epoch_init/derain", default_output_name, "mwirnet_5k_init", "output"),
            ColumnSpec("MWIR-Net-TTA", ROOT / "outputs/mwirnet_output_final_tta_multisplit/derain", default_output_name, "mwirnet_tta", "output"),
        ],
    )

    rain100l_ablation = GroupSpec(
        name="rain100l_ablation",
        title="Rain100L ablation comparison",
        input_dir=ROOT / "test/derain/Rain100L/input",
        target_dir=ROOT / "test/derain/Rain100L/target",
        samples=[make_sample(f"{i}.png") for i in [1, 25, 100]],
        output_columns=[
            ColumnSpec("Zero Prompt", ROOT / "outputs/mwirnet_output_ablation_zero_prompt/derain", default_output_name, "zero_prompt", "output"),
            ColumnSpec("No Channel Attention", ROOT / "outputs/mwirnet_output_ablation_no_channel_attention/derain", default_output_name, "no_channel_attention", "output"),
        ],
    )

    sots_outdoor = GroupSpec(
        name="sots_outdoor",
        title="SOTS outdoor main comparison",
        input_dir=ROOT / "test/dehaze/outdoor/input",
        target_dir=ROOT / "test/dehaze/outdoor/target",
        samples=sots_samples,
        output_columns=[
            ColumnSpec("PromptIR-official", PROMPTIR / "official_output/dehaze_outdoor", default_output_name, "promptir_official", "output"),
            ColumnSpec("AirNet-official-All", ROOT / "outputs/airnet_official_all/dehaze_outdoor", default_output_name, "airnet_official_all", "output"),
            ColumnSpec("PromptIR-5k-init", PROMPTIR / "promptir_output_5k_12epoch_init/dehaze_outdoor", default_output_name, "promptir_5k_init", "output"),
            ColumnSpec("AirNet-5k", ROOT / "outputs/airnet_5k_12epoch/dehaze_outdoor", default_output_name, "airnet_5k", "output"),
            ColumnSpec("MWIR-Net-5k-init", ROOT / "outputs/mwirnet_output_5k_12epoch_init/dehaze_outdoor", default_output_name, "mwirnet_5k_init", "output"),
            ColumnSpec("MWIR-Net-TTA", ROOT / "outputs/mwirnet_output_stage2_charb_edge002_tta_dehaze/dehaze_outdoor", default_output_name, "mwirnet_tta", "output"),
        ],
    )

    rain100h = GroupSpec(
        name="rain100h",
        title="Rain100H deraining comparison",
        input_dir=ROOT / "test/derain/Rain100H/input",
        target_dir=ROOT / "test/derain/Rain100H/target",
        samples=rain100h_samples,
        output_columns=[
            ColumnSpec("MWIR-Net Plain", ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Rain100H", default_output_name, "mwirnet_plain", "output"),
            ColumnSpec("MWIR-Net TTA", ROOT / "outputs/mwirnet_output_final_tta_multisplit/derain_Rain100H", default_output_name, "mwirnet_tta", "output"),
        ],
    )

    test1200 = GroupSpec(
        name="test1200",
        title="Test1200 deraining comparison",
        input_dir=ROOT / "test/derain/Test1200/input",
        target_dir=ROOT / "test/derain/Test1200/target",
        samples=test1200_samples,
        output_columns=[
            ColumnSpec("MWIR-Net", ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Test1200", default_output_name, "mwirnet_plain", "output"),
        ],
    )

    test2800 = GroupSpec(
        name="test2800",
        title="Test2800 deraining comparison",
        input_dir=ROOT / "test/derain/Test2800/input",
        target_dir=ROOT / "test/derain/Test2800/target",
        samples=test2800_samples,
        output_columns=[
            ColumnSpec("MWIR-Net", ROOT / "outputs/mwirnet_output_final_plain_multisplit/derain_Test2800", default_output_name, "mwirnet_plain", "output"),
        ],
    )

    nyuhaze500 = GroupSpec(
        name="nyuhaze500",
        title="NYU-Haze500 dehazing comparison",
        input_dir=ROOT / "test/dehaze/nyuhaze500/input",
        target_dir=ROOT / "test/dehaze/nyuhaze500/target",
        samples=nyu_samples,
        output_columns=[
            ColumnSpec("MWIR-Net", ROOT / "outputs/mwirnet_output_final_plain_multisplit_dehaze/dehaze_nyuhaze500", default_output_name, "mwirnet_plain", "output"),
        ],
    )

    return [
        rain100l_main,
        rain100l_ablation,
        sots_outdoor,
        rain100h,
        test1200,
        test2800,
        nyuhaze500,
    ]


def main() -> None:
    groups = group_specs()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for group in groups:
        all_rows.extend(export_group(group))

    write_text(OUT_ROOT / "README.md", build_readme(groups))
    write_text(OUT_ROOT / "manifest.csv", build_manifest_csv(all_rows))

    for path in [OUT_ROOT / "README.md", OUT_ROOT / "manifest.csv"]:
        print(path)


if __name__ == "__main__":
    main()
