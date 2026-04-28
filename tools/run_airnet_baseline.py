#!/usr/bin/env python3
"""Run an AirNet checkpoint on MWIR-Net Rain100L and SOTS outdoor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parents[1]
AIRNET_ROOT = Path("/root/autodl-tmp/2022-CVPR-AirNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AirNet weights on MWIR-Net test splits.")
    parser.add_argument("--airnet_root", type=Path, default=AIRNET_ROOT)
    parser.add_argument("--ckpt", type=Path, default=AIRNET_ROOT / "ckpt/All.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/airnet_official_all")
    parser.add_argument("--cuda", type=int, default=0)
    return parser.parse_args()


def load_airnet_state(ckpt: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        return checkpoint["net"]
    return checkpoint


def infer_batch_size(state: dict[str, torch.Tensor], default: int = 5) -> int:
    queue = state.get("E.E.queue")
    if queue is None or queue.ndim != 2:
        return default
    dim, queue_size = queue.shape
    if dim <= 0 or queue_size % dim != 0:
        return default
    return queue_size // dim


def run_split(net, dataset, task: str, output_dir: Path, airnet_utils) -> tuple[float, float]:
    dataset.set_dataset(task)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    psnr_meter = airnet_utils.AverageMeter()
    ssim_meter = airnet_utils.AverageMeter()

    with torch.no_grad():
        for ([name], degraded, clean) in tqdm(loader, desc=f"AirNet {task}"):
            degraded = degraded.cuda()
            clean = clean.cuda()
            restored = net(x_query=degraded, x_key=degraded)
            psnr, ssim, count = compute_psnr_ssim(restored, clean)
            psnr_meter.update(psnr, count)
            ssim_meter.update(ssim, count)
            airnet_utils.save_image_tensor(restored, str(output_dir / f"{name[0]}.png"))

    return psnr_meter.avg, ssim_meter.avg


def compute_psnr_ssim(restored: torch.Tensor, clean: torch.Tensor) -> tuple[float, float, int]:
    restored_np = torch.clamp(restored, 0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)
    clean_np = torch.clamp(clean, 0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)
    psnr = 0.0
    ssim = 0.0
    for idx in range(restored_np.shape[0]):
        psnr += peak_signal_noise_ratio(clean_np[idx], restored_np[idx], data_range=1.0)
        ssim += structural_similarity(clean_np[idx], restored_np[idx], data_range=1.0, channel_axis=-1)
    return psnr / restored_np.shape[0], ssim / restored_np.shape[0], restored_np.shape[0]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.airnet_root))

    from net.model import AirNet
    from utils.dataset_utils import DerainDehazeDataset
    from utils.image_io import save_image_tensor
    from utils.val_utils import AverageMeter

    airnet_utils = SimpleNamespace(
        AverageMeter=AverageMeter,
        save_image_tensor=save_image_tensor,
    )

    torch.manual_seed(0)
    torch.cuda.set_device(args.cuda)
    state = load_airnet_state(args.ckpt)
    batch_size = infer_batch_size(state)
    opt = SimpleNamespace(
        cuda=args.cuda,
        mode=3,
        batch_size=batch_size,
        derain_path=str(ROOT / "test/derain/Rain100L") + "/",
        dehaze_path=str(ROOT / "test/dehaze/outdoor") + "/",
    )

    net = AirNet(opt).cuda().eval()
    net.load_state_dict(state)
    dataset = DerainDehazeDataset(opt)

    derain_psnr, derain_ssim = run_split(net, dataset, "derain", args.output / "derain", airnet_utils)
    dehaze_psnr, dehaze_ssim = run_split(net, dataset, "dehaze", args.output / "dehaze_outdoor", airnet_utils)

    print(f"AirNet all-in-one Rain100L PSNR: {derain_psnr:.2f}, SSIM: {derain_ssim:.4f}")
    print(f"AirNet all-in-one SOTS outdoor PSNR: {dehaze_psnr:.2f}, SSIM: {dehaze_ssim:.4f}")


if __name__ == "__main__":
    main()
