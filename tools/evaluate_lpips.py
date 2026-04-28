import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS for restoration outputs.")
    parser.add_argument("--pred_dir", required=True, type=Path, help="directory containing restored images")
    parser.add_argument("--target_dir", required=True, type=Path, help="directory containing ground-truth images")
    parser.add_argument("--mode", choices=["derain", "dehaze"], required=True)
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_images(path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def build_target_index(target_dir):
    target_index = {}
    for path in list_images(target_dir):
        target_index.setdefault(path.stem, path)
    return target_index


def target_for_prediction(pred_path, target_dir, mode, target_index):
    if mode == "derain":
        exact = target_dir / pred_path.name
        if exact.exists():
            return exact
        return target_index.get(pred_path.stem, exact)

    # SOTS outdoor predictions keep the hazy filename, e.g. 0001_0.8_0.2.png,
    # while targets use only the clean image id, e.g. 0001.png.
    clean_id = pred_path.stem.split("_")[0]
    exact = target_dir / f"{clean_id}.png"
    if exact.exists():
        return exact
    return target_index.get(clean_id, exact)


def crop_to_base(image, base=16):
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


def center_crop_array(array, height, width):
    top = max(0, (array.shape[0] - height) // 2)
    left = max(0, (array.shape[1] - width) // 2)
    return array[top:top + height, left:left + width, :]


def crop_pair_to_common_size(pred_image, target_image, base=16):
    pred = np.array(pred_image)
    target = np.array(target_image)
    height = min(pred.shape[0], target.shape[0])
    width = min(pred.shape[1], target.shape[1])
    height = height - height % base
    width = width - width % base
    pred = center_crop_array(pred, height, width)
    target = center_crop_array(target, height, width)
    return Image.fromarray(pred), Image.fromarray(target)


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    image = crop_to_base(image, base=16)
    return transform(image).unsqueeze(0)


def load_image_pair(pred_path, target_path, transform):
    pred = Image.open(pred_path).convert("RGB")
    target = Image.open(target_path).convert("RGB")
    pred, target = crop_pair_to_common_size(pred, target, base=16)
    return transform(pred).unsqueeze(0), transform(target).unsqueeze(0)


def main():
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    model = lpips.LPIPS(net=args.net).to(args.device)
    model.eval()

    pred_paths = list_images(args.pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No images found in {args.pred_dir}")
    target_index = build_target_index(args.target_dir)
    if not target_index:
        raise FileNotFoundError(f"No target images found in {args.target_dir}")

    scores = []
    missing = []
    with torch.no_grad():
        for pred_path in tqdm(pred_paths, desc=f"LPIPS {args.pred_dir.name}"):
            target_path = target_for_prediction(pred_path, args.target_dir, args.mode, target_index)
            if not target_path.exists():
                missing.append((pred_path.name, target_path.name))
                continue

            pred, target = load_image_pair(pred_path, target_path, transform)
            pred = pred.to(args.device)
            target = target.to(args.device)
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {pred_path.name}: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
                )

            score = model(pred, target).item()
            scores.append(score)

    if missing:
        preview = ", ".join([f"{p}->{t}" for p, t in missing[:5]])
        raise FileNotFoundError(f"Missing {len(missing)} target files. First: {preview}")

    if not scores:
        raise RuntimeError("No LPIPS scores were computed.")

    avg = sum(scores) / len(scores)
    print(f"LPIPS({args.net}) {args.mode}: {avg:.6f} over {len(scores)} images")


if __name__ == "__main__":
    main()
