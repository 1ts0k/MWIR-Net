#!/usr/bin/env python3
"""Generate isolated thesis assets for the MWIR-Net thesis run."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "thesis-workbench-run"
FIG = RUN / "paper-output" / "figures"
SRC_FIG = FIG

BG = (255, 255, 255)
TEXT = (31, 41, 55)
BORDER = (203, 213, 225)


def configure_fonts() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def font(size: int = 18) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(18)
SMALL = font(14)


def add_box(ax, xy, width, height, label, color="#e2e8f0", edge="#334155", size=10):
    patch = Rectangle(xy, width, height, linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=size, color="#0f172a")


def add_arrow(ax, start, end, color="#475569"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color=color))


def diagram_canvas(path: Path, title: str):
    configure_fonts()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.2, 4.62, title, fontsize=13, weight="bold", color="#0f172a")
    return fig, ax


def transformer_block() -> None:
    fig, ax = diagram_canvas(FIG / "figure-3-2-transformer-block.png", "Transformer Restoration Block")
    add_box(ax, (0.4, 2.1), 1.0, 0.7, "Input", "#dbeafe")
    add_box(ax, (1.8, 2.1), 1.25, 0.7, "LayerNorm", "#e0f2fe")
    add_box(ax, (3.45, 2.1), 1.35, 0.7, "MDTA\nAttention", "#dcfce7")
    add_box(ax, (5.2, 2.1), 1.25, 0.7, "Residual\nAdd", "#fef3c7")
    add_box(ax, (6.85, 2.1), 1.25, 0.7, "LayerNorm", "#e0f2fe")
    add_box(ax, (8.5, 2.1), 1.35, 0.7, "GDFN\nFeed-forward", "#ede9fe")
    add_box(ax, (9.95, 2.1), 0.75, 0.7, "Add", "#fef3c7")
    for x1, x2 in [(1.4, 1.8), (3.05, 3.45), (4.8, 5.2), (6.45, 6.85), (8.1, 8.5), (9.85, 9.95)]:
        add_arrow(ax, (x1, 2.45), (x2, 2.45))
    add_arrow(ax, (0.9, 2.1), (5.8, 1.25))
    add_arrow(ax, (5.8, 1.25), (5.8, 2.1))
    add_arrow(ax, (5.8, 2.8), (10.35, 3.55))
    add_arrow(ax, (10.35, 3.55), (10.35, 2.8))
    ax.text(0.4, 0.45, "Evidence: net/mwirnet.py defines LayerNorm, Attention, FeedForward and TransformerBlock.", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "figure-3-2-transformer-block.png", bbox_inches="tight")
    plt.close(fig)


def prompt_module() -> None:
    fig, ax = diagram_canvas(FIG / "figure-3-3-prompt-module.png", "Weather-aware Prompt Module")
    add_box(ax, (0.3, 2.2), 1.35, 0.7, "Decoder\nFeature", "#dbeafe")
    add_box(ax, (2.0, 2.2), 1.2, 0.7, "Global\nPooling", "#e0f2fe")
    add_box(ax, (3.55, 2.2), 1.25, 0.7, "Linear\nSoftmax", "#dcfce7")
    add_box(ax, (5.15, 2.95), 1.45, 0.7, "Prompt\nDictionary", "#fef3c7")
    add_box(ax, (5.15, 1.45), 1.45, 0.7, "Weighted\nPrompt", "#fef3c7")
    add_box(ax, (7.0, 1.45), 1.25, 0.7, "3x3 Conv", "#ede9fe")
    add_box(ax, (8.65, 1.45), 1.25, 0.7, "Channel\nAttention", "#dcfce7")
    add_box(ax, (10.15, 1.45), 0.65, 0.7, "Fuse", "#fde68a")
    for x1, x2 in [(1.65, 2.0), (3.2, 3.55), (4.8, 5.15), (6.6, 7.0), (8.25, 8.65), (9.9, 10.15)]:
        add_arrow(ax, (x1, 2.55 if x1 < 5 else 1.8), (x2, 2.55 if x2 <= 5.15 else 1.8))
    add_arrow(ax, (5.85, 2.95), (5.85, 2.15))
    ax.text(0.3, 0.48, "Ablation modes: full, zero_prompt and no_channel_attention.", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "figure-3-3-prompt-module.png", bbox_inches="tight")
    plt.close(fig)


def tta_workflow() -> None:
    fig, ax = diagram_canvas(FIG / "figure-3-4-tta-workflow.png", "8-way TTA Self-ensemble Inference")
    add_box(ax, (0.4, 2.0), 1.15, 0.75, "Input", "#dbeafe")
    add_box(ax, (2.0, 2.0), 1.45, 0.75, "Rotate\n0/90/180/270", "#e0f2fe")
    add_box(ax, (3.9, 2.0), 1.35, 0.75, "Horizontal\nFlip", "#e0f2fe")
    add_box(ax, (5.75, 2.0), 1.35, 0.75, "MWIR-Net\nRestore", "#dcfce7")
    add_box(ax, (7.55, 2.0), 1.35, 0.75, "Inverse\nTransform", "#fef3c7")
    add_box(ax, (9.35, 2.0), 1.2, 0.75, "Mean\nFusion", "#fde68a")
    for x1, x2 in [(1.55, 2.0), (3.45, 3.9), (5.25, 5.75), (7.1, 7.55), (8.9, 9.35)]:
        add_arrow(ax, (x1, 2.38), (x2, 2.38))
    ax.text(0.4, 0.55, "Evidence: test.py restores four rotations and their flipped versions, then averages eight outputs.", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "figure-3-4-tta-workflow.png", bbox_inches="tight")
    plt.close(fig)


def experiment_workflow() -> None:
    fig, ax = diagram_canvas(FIG / "figure-4-1-experiment-workflow.png", "Training, Inference and Evaluation Workflow")
    labels = [
        ("Datasets\nRAIN13K/OTS/SOTS", "#dbeafe"),
        ("prepare_mwir_data.py\nsoft links + lists", "#e0f2fe"),
        ("MWIR-Net training\nAdamW + warmup cosine", "#dcfce7"),
        ("Checkpoints\nstage1/stage2", "#fef3c7"),
        ("test.py\nplain/TTA inference", "#fde68a"),
        ("Metric recompute\nPSNR/SSIM/LPIPS", "#ede9fe"),
    ]
    x = 0.35
    for label, color in labels:
        add_box(ax, (x, 2.1), 1.45, 0.78, label, color, size=8.8)
        if x < 8.7:
            add_arrow(ax, (x + 1.45, 2.49), (x + 1.75, 2.49))
        x += 1.75
    ax.text(0.35, 0.6, "Evidence: tools/prepare_mwir_data.py, train.py, test.py and docs/metric summary.", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "figure-4-1-experiment-workflow.png", bbox_inches="tight")
    plt.close(fig)


def fit_image(path: Path, size=(250, 160)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 12, size[1] - 12), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, BG)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def draw_center(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=TEXT):
    x, y, w, h = box
    b = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x + (w - (b[2] - b[0])) / 2, y + (h - (b[3] - b[1])) / 2), text, font=fnt, fill=fill)


def make_grid(path: Path, rows: list[tuple[str, list[Path]]], columns: list[str], cell=(250, 160)) -> None:
    header_h = 42
    label_h = 24
    width = len(columns) * cell[0]
    height = header_h + len(rows) * (cell[1] + label_h)
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for i, col in enumerate(columns):
        x = i * cell[0]
        draw.rectangle([x, 0, x + cell[0] - 1, header_h - 1], fill=(235, 245, 255), outline=BORDER)
        draw_center(draw, (x, 0, cell[0], header_h), col, FONT)
    for r, (label, paths) in enumerate(rows):
        y = header_h + r * (cell[1] + label_h)
        for c, image_path in enumerate(paths):
            x = c * cell[0]
            canvas.paste(fit_image(image_path, cell), (x, y))
            draw.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=BORDER)
        draw.rectangle([0, y + cell[1], width - 1, y + cell[1] + label_h - 1], fill=(248, 250, 252), outline=BORDER)
        draw_center(draw, (0, y + cell[1], width, label_h), label, SMALL)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def degradation_examples() -> None:
    rows = [
        (
            "Rain100L sample",
            [
                ROOT / "test/derain/Rain100L/input/1.png",
                ROOT / "test/derain/Rain100L/target/1.png",
            ],
        ),
        (
            "SOTS outdoor sample",
            [
                ROOT / "test/dehaze/outdoor/input/0001_0.8_0.2.jpg",
                ROOT / "test/dehaze/outdoor/target/0001.png",
            ],
        ),
    ]
    make_grid(FIG / "figure-2-1-degradation-examples.png", rows, ["Degraded input", "Clean target"], cell=(310, 205))


def rain_visual() -> None:
    rows = []
    for name in ["1.png", "10.png", "50.png"]:
        rows.append(
            (
                Path(name).stem,
                [
                    ROOT / "test/derain/Rain100L/input" / name,
                    ROOT / "outputs/restormer_official_derain_rain100l/Deraining" / name,
                    Path("/root/autodl-tmp/PromptIR/official_output/derain") / name,
                    ROOT / "outputs/mwirnet_output_stage2_charb_edge002_tta/derain" / name,
                    ROOT / "test/derain/Rain100L/target" / name,
                ],
            )
        )
    make_grid(FIG / "figure-5-1-rain-visual.png", rows, ["Input", "Restormer", "PromptIR", "MWIR-Net", "GT"])


def haze_visual() -> None:
    rows = []
    for name in ["0001_0.8_0.2.png", "0030_0.95_0.12.png", "0100_0.9_0.12.png"]:
        stem = Path(name).stem
        clean_id = stem.split("_")[0]
        rows.append(
            (
                stem,
                [
                    ROOT / "test/dehaze/outdoor/input" / f"{stem}.jpg",
                    Path("/root/autodl-tmp/PromptIR/official_output/dehaze_outdoor") / name,
                    ROOT / "outputs/mwirnet_output_stage2_charb_edge002_tta_dehaze/dehaze_outdoor" / name,
                    ROOT / "test/dehaze/outdoor/target" / f"{clean_id}.png",
                ],
            )
        )
    make_grid(FIG / "figure-5-2-haze-visual.png", rows, ["Input", "PromptIR", "MWIR-Net", "GT"])


def bar_chart(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, colors=None) -> None:
    configure_fonts()
    fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=180)
    colors = colors or ["#60a5fa"] * len(values)
    ax.bar(range(len(values)), values, color=colors, edgecolor="#334155", linewidth=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", color="#e2e8f0")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def metric_charts() -> None:
    bar_chart(
        FIG / "figure-5-3-rain100l-psnr.png",
        ["Median", "AirNet", "MWIR-Net", "MPRNet", "PromptIR", "Restormer"],
        [24.38, 34.90, 33.08, 34.95, 37.32, 37.57],
        "Rain100L PSNR Comparison",
        "PSNR (dB)",
        ["#cbd5e1", "#93c5fd", "#22c55e", "#a78bfa", "#f59e0b", "#ef4444"],
    )
    bar_chart(
        FIG / "figure-5-4-sots-psnr.png",
        ["CLAHE", "AirNet", "PromptIR", "MWIR-Net"],
        [16.30, 27.68, 30.35, 32.04],
        "SOTS Outdoor PSNR Comparison",
        "PSNR (dB)",
        ["#cbd5e1", "#93c5fd", "#f59e0b", "#22c55e"],
    )
    bar_chart(
        FIG / "figure-5-5-derain-multisplit.png",
        ["GT-RAIN", "Test100", "Rain100H", "Test1200", "Test2800", "Rain100L"],
        [21.03, 23.77, 25.05, 30.03, 30.66, 33.08],
        "MWIR-Net Deraining Multi-split PSNR",
        "PSNR (dB)",
        ["#93c5fd", "#93c5fd", "#93c5fd", "#22c55e", "#22c55e", "#22c55e"],
    )


def training_loss() -> None:
    configure_fonts()
    curves = [
        ("stage1 5k+5k", ROOT / "logs/mwirnet/version_0/metrics.csv"),
        ("dehaze stage2", ROOT / "logs/mwirnet/version_10/metrics.csv"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=180)
    for label, path in curves:
        steps, loss = [], []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("train_loss") and row.get("step"):
                    steps.append(int(float(row["step"])))
                    loss.append(float(row["train_loss"]))
        ax.plot(steps, loss, marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_title("Training Loss Curves", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.grid(color="#e2e8f0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure-4-3-training-loss.png", bbox_inches="tight")
    plt.close(fig)


def ablation_chart() -> None:
    configure_fonts()
    labels = ["zero_prompt", "no_channel_attention"]
    psnr = [32.7213, 32.7302]
    psnr_std = [0.0404, 0.0425]
    ssim = [0.941258, 0.941250]
    ssim_std = [0.000583, 0.000573]
    lpips = [0.087695, 0.087472]
    lpips_std = [0.001282, 0.001323]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=180)
    for ax, vals, errs, title, ylabel in [
        (axes[0], psnr, psnr_std, "PSNR", "dB"),
        (axes[1], ssim, ssim_std, "SSIM", "score"),
        (axes[2], lpips, lpips_std, "LPIPS", "distance"),
    ]:
        ax.bar(labels, vals, yerr=errs, color=["#60a5fa", "#22c55e"], edgecolor="#334155", capsize=4)
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", color="#e2e8f0")
    fig.suptitle("3-seed Fair Ablation on Rain100L", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figure-5-6-ablation.png", bbox_inches="tight")
    plt.close(fig)


def copy_base_figures() -> None:
    copies = {
        "mwirnet_architecture.png": "figure-3-1-mwirnet-architecture.png",
        "gtrain_real_rain_examples.png": "figure-4-2-gtrain-examples.png",
    }
    for src, dst in copies.items():
        source = SRC_FIG / src
        target = FIG / dst
        if source.exists() and source != target:
            shutil.copy2(source, target)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    copy_base_figures()
    degradation_examples()
    transformer_block()
    prompt_module()
    tta_workflow()
    experiment_workflow()
    training_loss()
    rain_visual()
    haze_visual()
    metric_charts()
    ablation_chart()
    print(f"Generated assets in {FIG}")


if __name__ == "__main__":
    main()
