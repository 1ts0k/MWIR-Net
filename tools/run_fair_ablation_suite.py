#!/usr/bin/env python3
"""Run the Rain100L fair ablation suite end to end.

This script trains the three MWIR-Net ablation modes across multiple seeds,
generates Rain100L predictions, evaluates PSNR/SSIM/LPIPS on the saved images,
and writes per-run plus aggregated summaries.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import evaluate_baseline_outputs as baseline_eval  # noqa: E402


DEFAULT_INIT_CKPT = PROJECT_ROOT / "checkpoints/mwirnet_ckpt_5k_12epoch_init/epoch=11-step=3744.ckpt"


@dataclass
class RunResult:
    mode: str
    seed: int
    split: str
    images: int
    psnr: float
    ssim: float
    lpips: float
    ckpt_path: Path
    pred_dir: Path
    train_log: Path
    test_log: Path
    eval_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MWIR-Net fair ablation suite.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="training seeds")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "zero_prompt", "no_channel_attention"],
        choices=["full", "zero_prompt", "no_channel_attention"],
        help="ablation modes to run",
    )
    parser.add_argument(
        "--derain-splits",
        nargs="+",
        default=["Rain100L"],
        help="derain test splits to evaluate",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--max-derain", type=int, default=5000)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.02)
    parser.add_argument("--pixel-loss-type", type=str, default="charbonnier", choices=["l1", "charbonnier"])
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--run-id", type=str, default=None, help="stable run name; defaults to timestamp")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/fair_ablation_suite",
        help="base directory for generated predictions",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/fair_ablation_suite",
        help="base directory for checkpoints",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=PROJECT_ROOT / "logs/fair_ablation_suite",
        help="base directory for logs and summaries",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=PROJECT_ROOT / "test/derain",
        help="root directory containing GT test splits",
    )
    return parser.parse_args()


def ensure_root(run_root: Path, *parts: str) -> Path:
    path = run_root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_to_pred_subdir(split: str) -> str:
    return "derain" if split == "Rain100L" else f"derain_{split}"


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    log: {log_path}")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def evaluate_split(
    pred_dir: Path,
    target_dir: Path,
    device: str,
    lpips_model: lpips.LPIPS,
) -> tuple[int, float, float, float]:
    pred_paths = baseline_eval.list_images(pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No images found in {pred_dir}")

    if not baseline_eval.list_images(target_dir):
        raise FileNotFoundError(f"No target images found in {target_dir}")

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    lpips_scores: list[float] = []
    missing: list[tuple[str, str]] = []

    from torchvision import transforms

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    with torch.no_grad():
        for pred_path in tqdm(pred_paths, desc=f"Eval {pred_dir.name}"):
            target_path = baseline_eval.target_for_prediction(pred_path, target_dir, "derain")
            if not target_path.exists():
                missing.append((pred_path.name, target_path.name))
                continue

            pred, target = baseline_eval.load_pair(pred_path, target_path)
            pred_float = pred.astype(np.float32) / 255.0
            target_float = target.astype(np.float32) / 255.0
            psnr_scores.append(peak_signal_noise_ratio(target_float, pred_float, data_range=1.0))
            ssim_scores.append(
                structural_similarity(target_float, pred_float, data_range=1.0, channel_axis=-1)
            )

            pred_tensor = to_tensor(Image.fromarray(pred)).unsqueeze(0).to(device)
            target_tensor = to_tensor(Image.fromarray(target)).unsqueeze(0).to(device)
            lpips_scores.append(lpips_model(pred_tensor, target_tensor).item())

    if missing:
        preview = ", ".join(f"{p}->{t}" for p, t in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} target files. First: {preview}")

    return (
        len(pred_paths),
        float(sum(psnr_scores) / len(psnr_scores)),
        float(sum(ssim_scores) / len(ssim_scores)),
        float(sum(lpips_scores) / len(lpips_scores)),
    )


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, results: list[RunResult], grouped: dict[str, dict[str, tuple[float, float]]]) -> None:
    lines = [
        "# Fair Ablation Summary",
        "",
        "| Mode | Metric | Mean | Std |",
        "|---|---|---:|---:|",
    ]
    for mode in ["full", "zero_prompt", "no_channel_attention"]:
        if mode not in grouped:
            continue
        for metric in ["psnr", "ssim", "lpips"]:
            mean_val, std_val = grouped[mode][metric]
            lines.append(f"| {mode} | {metric.upper()} | {mean_val:.6f} | {std_val:.6f} |")

    lines += [
        "",
        "## Per-run Results",
        "",
        "| Mode | Seed | Split | Images | PSNR | SSIM | LPIPS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.mode} | {row.seed} | {row.split} | {row.images} | {row.psnr:.4f} | {row.ssim:.4f} | {row.lpips:.6f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_output_root = args.output_root / run_id
    run_ckpt_root = args.ckpt_root / run_id
    run_log_root = args.log_root / run_id
    run_output_root.mkdir(parents=True, exist_ok=True)
    run_ckpt_root.mkdir(parents=True, exist_ok=True)
    run_log_root.mkdir(parents=True, exist_ok=True)

    eval_device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(eval_device).eval()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Run id: {run_id}")
    print(f"Output root: {run_output_root}")
    print(f"Checkpoint root: {run_ckpt_root}")
    print(f"Log root: {run_log_root}")

    results: list[RunResult] = []

    for mode in args.modes:
        for seed in args.seeds:
            seed_name = f"seed_{seed}"
            ckpt_dir = ensure_root(run_ckpt_root, mode, seed_name)
            pred_root = ensure_root(run_output_root, mode, seed_name)
            run_logs = ensure_root(run_log_root, mode, seed_name)

            train_log = run_logs / "train.log"
            test_log = run_logs / "test.log"
            eval_log = run_logs / "eval.log"

            train_cmd = [
                sys.executable,
                "train.py",
                "--cuda",
                str(args.cuda),
                "--de_type",
                "derain",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--patch_size",
                str(args.patch_size),
                "--num_workers",
                str(args.num_workers),
                "--lr",
                str(args.lr),
                "--warmup_epochs",
                str(args.warmup_epochs),
                "--max_derain",
                str(args.max_derain),
                "--subset_seed",
                str(args.subset_seed),
                "--seed",
                str(seed),
                "--edge_loss_weight",
                str(args.edge_loss_weight),
                "--pixel_loss_type",
                args.pixel_loss_type,
                "--charbonnier_eps",
                str(args.charbonnier_eps),
                "--init_ckpt",
                str(args.init_ckpt),
                "--ablation_mode",
                mode,
                "--ckpt_dir",
                str(ckpt_dir),
                "--csv_log_dir",
                str(run_logs / "trainer_csv"),
                "--save_last_only",
                "--num_gpus",
                str(args.num_gpus),
                "--precision",
                args.precision,
                "--wblogger",
                "none",
            ]
            run_command(train_cmd, train_log)

            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = last_ckpt
            else:
                ckpts = sorted(ckpt_dir.glob(f"epoch={args.epochs - 1}-step=*.ckpt"))
                if not ckpts:
                    ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
                if not ckpts:
                    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
                ckpt_path = ckpts[-1]

            test_cmd = [
                sys.executable,
                "test.py",
                "--cuda",
                str(args.cuda),
                "--mode",
                "1",
                "--ckpt_path",
                str(ckpt_path),
                "--output_path",
                str(pred_root) + os.sep,
                "--derain_splits",
                *args.derain_splits,
                "--ablation_mode",
                mode,
            ]
            run_command(test_cmd, test_log)

            for split in args.derain_splits:
                pred_dir = pred_root / split_to_pred_subdir(split)
                target_dir = args.target_root / split / "target"
                if not target_dir.exists():
                    raise FileNotFoundError(f"Missing target directory: {target_dir}")

                images, psnr, ssim, lpips_score = evaluate_split(
                    pred_dir=pred_dir,
                    target_dir=target_dir,
                    device=eval_device,
                    lpips_model=lpips_model,
                )

                eval_log.parent.mkdir(parents=True, exist_ok=True)
                with eval_log.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{mode} seed={seed} split={split} images={images} PSNR={psnr:.4f} "
                        f"SSIM={ssim:.4f} LPIPS={lpips_score:.6f}\n"
                    )

                results.append(
                    RunResult(
                        mode=mode,
                        seed=seed,
                        split=split,
                        images=images,
                        psnr=psnr,
                        ssim=ssim,
                        lpips=lpips_score,
                        ckpt_path=ckpt_path,
                        pred_dir=pred_dir,
                        train_log=train_log,
                        test_log=test_log,
                        eval_log=eval_log,
                    )
                )

                print(
                    f"[{mode} | seed {seed} | {split}] PSNR={psnr:.4f} "
                    f"SSIM={ssim:.4f} LPIPS={lpips_score:.6f} ({images} images)"
                )

    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for mode in args.modes:
        mode_rows = [r for r in results if r.mode == mode]
        grouped[mode] = {
            "psnr": mean_std(r.psnr for r in mode_rows),
            "ssim": mean_std(r.ssim for r in mode_rows),
            "lpips": mean_std(r.lpips for r in mode_rows),
        }

    results_csv = run_log_root / "per_run_metrics.csv"
    summary_csv = run_log_root / "summary.csv"
    summary_md = run_log_root / "summary.md"

    write_csv(
        results_csv,
        [
            {
                "mode": r.mode,
                "seed": r.seed,
                "split": r.split,
                "images": r.images,
                "psnr": f"{r.psnr:.6f}",
                "ssim": f"{r.ssim:.6f}",
                "lpips": f"{r.lpips:.6f}",
                "ckpt_path": str(r.ckpt_path),
                "pred_dir": str(r.pred_dir),
                "train_log": str(r.train_log),
                "test_log": str(r.test_log),
                "eval_log": str(r.eval_log),
            }
            for r in results
        ],
    )

    summary_rows = []
    for mode in args.modes:
        summary_rows.append(
            {
                "mode": mode,
                "psnr_mean": f"{grouped[mode]['psnr'][0]:.6f}",
                "psnr_std": f"{grouped[mode]['psnr'][1]:.6f}",
                "ssim_mean": f"{grouped[mode]['ssim'][0]:.6f}",
                "ssim_std": f"{grouped[mode]['ssim'][1]:.6f}",
                "lpips_mean": f"{grouped[mode]['lpips'][0]:.6f}",
                "lpips_std": f"{grouped[mode]['lpips'][1]:.6f}",
            }
        )
    write_csv(summary_csv, summary_rows)
    write_markdown_summary(summary_md, results, grouped)

    print("\nSummary")
    for mode in args.modes:
        psnr_mean, psnr_std = grouped[mode]["psnr"]
        ssim_mean, ssim_std = grouped[mode]["ssim"]
        lpips_mean, lpips_std = grouped[mode]["lpips"]
        print(
            f"{mode}: PSNR {psnr_mean:.4f}±{psnr_std:.4f}, "
            f"SSIM {ssim_mean:.4f}±{ssim_std:.4f}, "
            f"LPIPS {lpips_mean:.6f}±{lpips_std:.6f}"
        )
    print(f"\nPer-run metrics: {results_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD: {summary_md}")


if __name__ == "__main__":
    main()
