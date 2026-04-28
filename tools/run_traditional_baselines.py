#!/usr/bin/env python3
"""Run simple traditional restoration baselines.

These baselines are intentionally lightweight and are meant to provide weak
classical references for thesis discussion, not state-of-the-art restoration.
Supported methods:

- median: 3x3 median filtering, mainly for deraining.
- clahe: contrast-limited histogram equalization on luminance.
- retinex: single-scale Retinex-style luminance enhancement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import cv2
except Exception:
    cv2 = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def crop_to_base(image: Image.Image, base: int = 16) -> Image.Image:
    array = np.array(image)
    h, w = array.shape[:2]
    crop_h = h % base
    crop_w = w % base
    array = array[
        crop_h // 2 : h - crop_h + crop_h // 2,
        crop_w // 2 : w - crop_w + crop_w // 2,
        :,
    ]
    return Image.fromarray(array)


def apply_median(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(size=3))


def apply_clahe(image: Image.Image) -> Image.Image:
    if cv2 is None:
        return ImageOps.autocontrast(ImageOps.equalize(image))

    rgb = np.array(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def apply_retinex(image: Image.Image, sigma: float = 30.0) -> Image.Image:
    if cv2 is None:
        return ImageOps.autocontrast(image)

    rgb = np.asarray(image).astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    retinex = np.log(rgb) - np.log(blur)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
    retinex = np.clip(retinex * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(retinex)


def restore(image: Image.Image, method: str) -> Image.Image:
    if method == "median":
        return apply_median(image)
    if method == "clahe":
        return apply_clahe(image)
    if method == "retinex":
        return apply_retinex(image)
    raise ValueError(f"Unknown method: {method}")


def target_for_prediction(input_path: Path, target_dir: Path, task: str) -> Path:
    if task == "derain":
        return target_dir / input_path.name

    clean_id = input_path.stem.split("_")[0]
    return target_dir / f"{clean_id}.png"


def evaluate_pair(pred: Image.Image, target: Image.Image) -> tuple[float, float]:
    pred_arr = np.asarray(crop_to_base(pred)).astype(np.float32) / 255.0
    target_arr = np.asarray(crop_to_base(target)).astype(np.float32) / 255.0
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch: {pred_arr.shape} vs {target_arr.shape}")
    psnr = peak_signal_noise_ratio(target_arr, pred_arr, data_range=1.0)
    try:
        ssim = structural_similarity(target_arr, pred_arr, data_range=1.0, channel_axis=-1)
    except TypeError:
        ssim = structural_similarity(target_arr, pred_arr, data_range=1.0, multichannel=True)
    return psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple traditional baselines.")
    parser.add_argument("--task", choices=["derain", "dehaze"], required=True)
    parser.add_argument("--method", choices=["median", "clahe", "retinex"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = list_images(args.input_dir)
    if not inputs:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    psnr_values = []
    ssim_values = []
    for input_path in inputs:
        image = Image.open(input_path).convert("RGB")
        restored = restore(image, args.method)
        restored.save(args.output_dir / input_path.name)

        if args.target_dir is not None:
            target_path = target_for_prediction(input_path, args.target_dir, args.task)
            if target_path.exists():
                target = Image.open(target_path).convert("RGB")
                psnr, ssim = evaluate_pair(restored, target)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

    print(f"{args.method} {args.task}: saved {len(inputs)} images to {args.output_dir}")
    if psnr_values:
        print(
            "PSNR: {:.2f}, SSIM: {:.4f} over {} images".format(
                float(np.mean(psnr_values)),
                float(np.mean(ssim_values)),
                len(psnr_values),
            )
        )


if __name__ == "__main__":
    main()
