#!/usr/bin/env python3
"""Evaluate restored image folders with PSNR, SSIM, and optional LPIPS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a restored output directory.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--target_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["derain", "dehaze"], required=True)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def target_for_prediction(pred_path: Path, target_dir: Path, mode: str) -> Path:
    if mode == "derain":
        return target_dir / pred_path.name

    clean_id = pred_path.stem.split("_")[0]
    return target_dir / f"{clean_id}.png"


def center_crop(array: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max(0, (array.shape[0] - height) // 2)
    left = max(0, (array.shape[1] - width) // 2)
    return array[top : top + height, left : left + width, :]


def load_pair(pred_path: Path, target_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.array(Image.open(pred_path).convert("RGB"))
    target = np.array(Image.open(target_path).convert("RGB"))
    height = min(pred.shape[0], target.shape[0])
    width = min(pred.shape[1], target.shape[1])
    height -= height % 16
    width -= width % 16
    pred = center_crop(pred, height, width)
    target = center_crop(target, height, width)
    return pred, target


def main() -> None:
    args = parse_args()
    pred_paths = list_images(args.pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No images found in {args.pred_dir}")

    lpips_model = None
    transform = None
    lpips_scores: list[float] = []
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    missing: list[tuple[str, str]] = []

    with torch.no_grad():
        for pred_path in tqdm(pred_paths, desc=f"Eval {args.pred_dir.name}"):
            target_path = target_for_prediction(pred_path, args.target_dir, args.mode)
            if not target_path.exists():
                missing.append((pred_path.name, target_path.name))
                continue

            pred, target = load_pair(pred_path, target_path)
            pred_float = pred.astype(np.float32) / 255.0
            target_float = target.astype(np.float32) / 255.0
            psnr_scores.append(peak_signal_noise_ratio(target_float, pred_float, data_range=1.0))
            ssim_scores.append(
                structural_similarity(target_float, pred_float, data_range=1.0, channel_axis=-1)
            )

            if lpips_model is not None and transform is not None:
                pred_tensor = transform(Image.fromarray(pred)).unsqueeze(0).to(args.device)
                target_tensor = transform(Image.fromarray(target)).unsqueeze(0).to(args.device)
                lpips_scores.append(lpips_model(pred_tensor, target_tensor).item())

    if missing:
        preview = ", ".join(f"{p}->{t}" for p, t in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} target files. First: {preview}")

    result = (
        f"PSNR: {sum(psnr_scores) / len(psnr_scores):.2f}, "
        f"SSIM: {sum(ssim_scores) / len(ssim_scores):.4f} over {len(psnr_scores)} images"
    )
    if lpips_scores:
        result += f", LPIPS(alex): {sum(lpips_scores) / len(lpips_scores):.6f}"
    print(result)


if __name__ == "__main__":
    main()
