# MWIR-Net

Multi-scale Weather-aware Image Restoration Network for fog and rain degraded images.

This project focuses on two restoration tasks:

- Deraining
- Dehazing

本仓库用于毕业设计“基于深度学习在雾雨退化场景下的图像复原算法研究”。当前实验主线为：在公开图像复原权重初始化的基础上，进行去雨/去雾联合微调，再通过二阶段去雨微调、Charbonnier 损失、边缘一致性约束和 TTA 自集成推理提升复原效果。

## Model

MWIR-Net uses a multi-scale encoder-decoder restoration backbone with:

- Transformer restoration blocks
- Multi-scale feature fusion
- Weather-aware prompt blocks
- Prompt channel attention
- Optional Sobel edge consistency loss

The main model implementation is in:

```text
net/mwirnet.py
```

Training and evaluation entry points:

```text
train.py
test.py
tools/evaluate_lpips.py
tools/test_checkpoint_ensemble.py
tools/make_visual_comparisons.py
```

## Data Layout

Datasets are stored under:

```text
datasets/
├── RAIN13K
├── OTS_ALPHA
├── SOTS
├── ITS_v2
└── GT-RAIN
```

Generate training and test symlinks:

```bash
python tools/prepare_mwir_data.py --dehaze-source ots
```

## Train

Small fog/rain training:

```bash
python train.py \
  --de_type derain dehaze \
  --epochs 12 \
  --batch_size 32 \
  --patch_size 128 \
  --num_workers 8 \
  --num_gpus 1 \
  --precision 16-mixed \
  --wblogger none \
  --derain_repeat 1 \
  --max_derain 5000 \
  --max_dehaze 5000 \
  --edge_loss_weight 0.05 \
  --ckpt_dir checkpoints/mwirnet_ckpt_5k_12epoch
```

Fine-tuning from a compatible restoration checkpoint usually converges faster:

```bash
python train.py \
  --de_type derain dehaze \
  --epochs 12 \
  --batch_size 32 \
  --patch_size 128 \
  --num_workers 8 \
  --num_gpus 1 \
  --precision 16-mixed \
  --wblogger none \
  --derain_repeat 1 \
  --max_derain 5000 \
  --max_dehaze 5000 \
  --edge_loss_weight 0.05 \
  --init_ckpt /root/autodl-tmp/PromptIR/ckpt/model.ckpt \
  --ckpt_dir checkpoints/mwirnet_ckpt_5k_12epoch_init
```

Second-stage deraining fine-tuning with Charbonnier loss and edge consistency:

```bash
python train.py \
  --de_type derain \
  --epochs 8 \
  --batch_size 32 \
  --patch_size 128 \
  --num_workers 8 \
  --num_gpus 1 \
  --precision 16-mixed \
  --wblogger none \
  --derain_repeat 1 \
  --max_derain 5000 \
  --lr 1e-5 \
  --warmup_epochs 1 \
  --pixel_loss_type charbonnier \
  --edge_loss_weight 0.02 \
  --init_ckpt checkpoints/mwirnet_ckpt_5k_12epoch_init/epoch=11-step=3744.ckpt \
  --ckpt_dir checkpoints/mwirnet_ckpt_stage2_charb_edge002
```

## Test

Deraining:

```bash
python test.py \
  --mode 1 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output/ \
  --derain_splits Rain100L
```

Dehazing:

```bash
python test.py \
  --mode 2 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output/ \
  --dehaze_splits outdoor
```

TTA self-ensemble inference:

```bash
python test.py \
  --mode 1 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output_tta/ \
  --derain_splits Rain100L \
  --tta
```

Checkpoint ensemble inference:

```bash
python tools/test_checkpoint_ensemble.py \
  --mode 1 \
  --output_path outputs/mwirnet_output_ensemble_3ckpt_tta/ \
  --tta \
  --ckpt_paths \
    checkpoints/mwirnet_ckpt_derain_stage2/epoch=1-step=856.ckpt \
    checkpoints/mwirnet_ckpt_derain_stage2/epoch=6-step=2996.ckpt \
    checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --derain_splits Rain100L
```

LPIPS evaluation:

```bash
python tools/evaluate_lpips.py \
  --mode derain \
  --pred_dir outputs/mwirnet_output_tta/derain \
  --target_dir test/derain/Rain100L/target
```

## Experimental Results

LPIPS uses the AlexNet backbone. Lower LPIPS is better.

| Method | Rain100L PSNR / SSIM / LPIPS | SOTS outdoor PSNR / SSIM / LPIPS |
|---|---:|---:|
| PromptIR official checkpoint | 37.44 / 0.9786 / 0.016323 | 30.59 / 0.9779 / 0.013528 |
| AirNet from scratch, 5k+5k, 12 epochs | 23.31 / 0.7928 / 0.267186 | 23.74 / 0.9309 / 0.054032 |
| PromptIR from scratch, 5k+5k, 12 epochs | 24.19 / 0.7971 / 0.259113 | 26.59 / 0.9437 / 0.029323 |
| PromptIR initialized fine-tuning, 5k+5k, 12 epochs | 31.63 / 0.9377 / 0.090232 | 30.75 / 0.9792 / 0.012616 |
| MWIR-Net from scratch, 5k+5k, 12 epochs | 25.16 / 0.8126 / 0.259754 | 26.22 / 0.9464 / 0.029608 |
| MWIR-Net initialized fine-tuning, 5k+5k, 12 epochs | 32.64 / 0.9387 / 0.091427 | 30.43 / 0.9780 / 0.012968 |
| MWIR-Net Charbonnier + edge0.02 | 32.89 / 0.9399 / 0.091470 | 31.72 / 0.9793 / 0.010678 |
| MWIR-Net Charbonnier + edge0.02 + TTA | 33.33 / 0.9447 / 0.087578 | 32.07 / 0.9805 / 0.009871 |

Checkpoint ensemble was also tested, but it was not adopted as the final inference setting:

| Inference | Rain100L PSNR / SSIM / LPIPS | SOTS outdoor PSNR / SSIM / LPIPS |
|---|---:|---:|
| 3-checkpoint ensemble | 32.95 / 0.9402 / 0.094919 | 30.90 / 0.9738 / 0.012469 |
| 3-checkpoint ensemble + TTA | 33.30 / 0.9444 / 0.091291 | 31.17 / 0.9750 / 0.011790 |

The final recommended setting is the single `Charbonnier + edge0.02` checkpoint with TTA self-ensemble inference.

## Documentation

- `雾雨复原实验结果记录.md`: detailed experiment log and thesis wording suggestions.
- `毕设任务完成情况对照.md`: thesis task checklist and remaining work.
- `毕设任务完成情况对照.html`: rendered HTML version of the task checklist.

## Notes

The high-score experiments use public pretrained checkpoint initialization followed by fine-tuning. They should not be described as full training from scratch.
