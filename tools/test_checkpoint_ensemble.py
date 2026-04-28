import argparse
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net.mwirnet import MWIRNet
from utils.dataset_utils import DerainDehazeDataset
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim


class MWIRLitModel(pl.LightningModule):
    def __init__(self, ablation_mode="full"):
        super().__init__()
        self.net = MWIRNet(decoder=True, ablation_mode=ablation_mode)

    def forward(self, x):
        return self.net(x)


def restore_with_tta(net, degraded):
    restored_sum = torch.zeros_like(degraded)
    dims = (-2, -1)

    for k in range(4):
        aug = torch.rot90(degraded, k=k, dims=dims)
        restored = net(aug)
        restored_sum += torch.rot90(restored, k=-k, dims=dims)

        aug = torch.flip(torch.rot90(degraded, k=k, dims=dims), dims=(-1,))
        restored = net(aug)
        restored = torch.rot90(torch.flip(restored, dims=(-1,)), k=-k, dims=dims)
        restored_sum += restored

    return restored_sum / 8.0


def restore_ensemble(nets, degraded, use_tta):
    restored_sum = torch.zeros_like(degraded)
    for net in nets:
        if use_tta:
            restored_sum += restore_with_tta(net, degraded)
        else:
            restored_sum += net(degraded)
    return restored_sum / len(nets)


def load_models(ckpt_paths, device, ablation_mode):
    nets = []
    for ckpt_path in ckpt_paths:
        print(f"Loading checkpoint: {ckpt_path}")
        model = MWIRLitModel.load_from_checkpoint(
            str(ckpt_path),
            strict=False,
            ablation_mode=ablation_mode,
        ).to(device)
        model.eval()
        nets.append(model)
    return nets


def output_name_for_task(task, split):
    if task == "derain":
        return "derain" if split == "Rain100L" else f"derain_{split}"
    return f"dehaze_{split}"


def test_split(args, nets, task, split, device):
    if task == "derain":
        base_path = args.derain_path
        args.derain_path = os.path.join(base_path, split) + "/"
    else:
        base_path = args.dehaze_path
        args.dehaze_path = os.path.join(base_path, split) + "/"

    dataset = DerainDehazeDataset(args, task=task, addnoise=False, sigma=15)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    output_dir = Path(args.output_path) / output_name_for_task(task, split)
    output_dir.mkdir(parents=True, exist_ok=True)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(loader, desc=f"{task}:{split}"):
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            restored = restore_ensemble(nets, degrad_patch, args.tta)

            temp_psnr, temp_ssim, n = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, n)
            ssim.update(temp_ssim, n)
            save_image_tensor(restored, str(output_dir / f"{degraded_name[0]}.png"))

    print(f"{task}:{split} PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")
    return psnr.avg, ssim.avg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate output-averaged MWIR-Net checkpoint ensemble.")
    parser.add_argument("--ckpt_paths", nargs="+", required=True, type=Path)
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True,
                        help="1 for derain, 2 for dehaze, 3 for both")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--derain_path", type=str, default="test/derain/")
    parser.add_argument("--dehaze_path", type=str, default="test/dehaze/")
    parser.add_argument("--derain_splits", nargs="+", default=["Rain100L"])
    parser.add_argument("--dehaze_splits", nargs="+", default=["outdoor"])
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--ablation_mode", type=str, default="full",
                        choices=["full", "zero_prompt", "no_channel_attention"])
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    nets = load_models(args.ckpt_paths, device, args.ablation_mode)
    print(f"Loaded {len(nets)} checkpoints. TTA: {args.tta}")

    if args.mode in [1, 3]:
        for split in args.derain_splits:
            test_split(args, nets, "derain", split, device)
    if args.mode in [2, 3]:
        for split in args.dehaze_splits:
            test_split(args, nets, "dehaze", split, device)


if __name__ == "__main__":
    main()
