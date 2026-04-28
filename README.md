# MWIR-Net

MWIR-Net（Multi-scale Weather-aware Image Restoration Network）是一个面向雾、雨退化场景的图像复原实验项目，主要用于去雨（deraining）和去雾（dehazing）任务。项目围绕毕业设计“基于深度学习在雾雨退化场景下的图像复原算法研究”展开，包含模型实现、训练脚本、测试脚本、指标复算工具和论文实验材料。

当前实验主线是：以 Transformer 多尺度复原骨干为基础，引入 weather-aware prompt 模块和通道注意力，并结合 Charbonnier 损失、Sobel 边缘一致性约束、TTA 自集成推理等策略提升雾雨图像复原质量。

## 项目特点

- 支持去雨、去雾两类退化复原任务。
- 使用多尺度编码器-解码器结构和 Transformer restoration block。
- 在解码阶段注入天气感知 prompt，用于建模不同退化类型。
- 支持 `zero_prompt`、`no_channel_attention` 等消融实验。
- 提供训练、推理、TTA、自集成、LPIPS/PSNR/SSIM 评估和可视化对比脚本。

## 目录结构

```text
MWIR-Net/
├── net/mwirnet.py                 # MWIR-Net 模型结构
├── train.py                       # 训练入口
├── test.py                        # 标准测试入口
├── demo.py                        # 单图或目录推理入口
├── options.py                     # 训练参数
├── tools/                         # 数据准备、评估、可视化和实验脚本
├── utils/                         # 数据集、图像读写、损失和指标工具
├── data_dir/                      # 训练列表文件
├── docs/                          # 实验记录和指标汇总
├── visual_comparisons/            # 论文/答辩用可视化对比图
├── env.yml                        # Conda 环境
└── INSTALL.md                     # 简短安装与 smoke test 说明
```

以下目录只在本地使用，不上传到 GitHub：

```text
data/
datasets/
test/
checkpoints/
ckpt/
smoke_ckpt/
outputs/
output/
traditional_output/
logs/
```

## 环境安装

推荐使用 Conda：

```bash
conda env create -f env.yml
conda activate mwirnet
```

如果只需要快速查看项目结构和文档，不需要准备完整训练环境。

## 数据准备

原始数据集默认放在 `datasets/` 下，例如：

```text
datasets/
├── RAIN13K/
├── OTS_ALPHA/
├── SOTS/
├── ITS_v2/
└── GT-RAIN/
```

生成训练和测试所需的软链接：

```bash
python tools/prepare_mwir_data.py --dehaze-source ots
```

该脚本会根据 `datasets/` 中的数据生成本地 `data/`、`test/` 目录，并更新 `data_dir/` 下的列表文件。`data/` 和 `test/` 已被 `.gitignore` 忽略，避免误上传大规模数据。

## 训练

1 步 smoke test：

```bash
python train.py \
  --de_type derain dehaze \
  --epochs 1 \
  --max_steps 1 \
  --batch_size 1 \
  --patch_size 128 \
  --num_workers 2 \
  --num_gpus 1 \
  --precision 16-mixed \
  --wblogger none \
  --max_derain 2 \
  --max_dehaze 2 \
  --ckpt_dir smoke_ckpt
```

5k 去雨 + 5k 去雾联合训练示例：

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
  --max_derain 5000 \
  --max_dehaze 5000 \
  --edge_loss_weight 0.05 \
  --ckpt_dir checkpoints/mwirnet_ckpt_5k_12epoch
```

使用兼容权重初始化：

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
  --max_derain 5000 \
  --max_dehaze 5000 \
  --edge_loss_weight 0.05 \
  --init_ckpt /path/to/pretrained.ckpt \
  --ckpt_dir checkpoints/mwirnet_ckpt_5k_12epoch_init
```

二阶段去雨微调示例：

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
  --max_derain 5000 \
  --lr 1e-5 \
  --warmup_epochs 1 \
  --pixel_loss_type charbonnier \
  --edge_loss_weight 0.02 \
  --init_ckpt checkpoints/mwirnet_ckpt_5k_12epoch_init/epoch=11-step=3744.ckpt \
  --ckpt_dir checkpoints/mwirnet_ckpt_stage2_charb_edge002
```

## 测试与推理

去雨测试：

```bash
python test.py \
  --mode 1 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output/ \
  --derain_splits Rain100L
```

去雾测试：

```bash
python test.py \
  --mode 2 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output/ \
  --dehaze_splits outdoor
```

TTA 自集成推理：

```bash
python test.py \
  --mode 1 \
  --ckpt_path checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt \
  --output_path outputs/mwirnet_output_tta/ \
  --derain_splits Rain100L \
  --tta
```

单图或目录推理：

```bash
python demo.py \
  --test_path test/demo/ \
  --output_path output/demo/ \
  --ckpt_name model.ckpt
```

## 指标评估

复算 PSNR、SSIM 和 LPIPS：

```bash
python tools/evaluate_baseline_outputs.py \
  --mode derain \
  --pred_dir outputs/mwirnet_output_tta/derain \
  --target_dir test/derain/Rain100L/target \
  --lpips
```

单独计算 LPIPS：

```bash
python tools/evaluate_lpips.py \
  --mode dehaze \
  --pred_dir outputs/mwirnet_output_tta/dehaze_outdoor \
  --target_dir test/dehaze/outdoor/target
```

完整指标表见：

- [docs/所有模型指标汇总.md](docs/所有模型指标汇总.md)

## 当前实验结果

统计日期：2026-04-28。下表摘自当前工作区保存图像的统一复算结果。PSNR、SSIM 越高越好，LPIPS(alex) 越低越好。

| 任务 | 数据集 | 当前代表结果 | PSNR | SSIM | LPIPS(alex) |
|---|---|---|---:|---:|---:|
| 去雨 | Rain100L | MWIR-Net-final_tta_multisplit | 33.08 | 0.9442 | 0.087578 |
| 去雨 | Test1200 | MWIR-Net-final_plain_multisplit | 30.03 | 0.8702 | 0.090416 |
| 去雨 | Test2800 | MWIR-Net-final_plain_multisplit | 30.66 | 0.9078 | 0.057636 |
| 去雨 | GT-RAIN-test | MWIR-Net-gtrain_plain | 21.03 | 0.5963 | 0.293823 |
| 去雾 | SOTS outdoor | MWIR-Net-stage2_charb_edge002_tta_dehaze | 32.04 | 0.9804 | 0.009871 |
| 去雾 | nyuhaze500 | MWIR-Net-final_plain_multisplit_dehaze | 17.20 | 0.8239 | 0.101394 |

公开官方权重基线在部分数据集上仍然更强，例如 Rain100L 上 Restormer-official 的 PSNR 为 37.57，PromptIR-official 的 SSIM 为 0.9778、LPIPS 为 0.016323。MWIR-Net 的主要贡献在于围绕雾雨联合复原任务完成了可复现实验链路、结构改造、损失函数和推理策略对比。

## 可视化对比

PPT 使用的对比图保存在 `visual_comparisons/ppt_5k_protocol/`。其中保留了 Rain100L 和 SOTS outdoor 的网格图与单样本横向对比条。

Rain100L 去雨对比：

![Rain100L PPT comparison](visual_comparisons/ppt_5k_protocol/rain100l_ppt_grid.png)

SOTS outdoor 去雾对比：

![SOTS outdoor PPT comparison](visual_comparisons/ppt_5k_protocol/sots_outdoor_ppt_grid.png)

## 文档

- [docs/所有模型指标汇总.md](docs/所有模型指标汇总.md)：全部模型输出指标汇总。
- [docs/基于深度学习在雾雨退化场景下的图像复原算法研究.txt](docs/基于深度学习在雾雨退化场景下的图像复原算法研究.txt)：论文正文材料。
- [INSTALL.md](INSTALL.md)：环境和 smoke test 简要说明。

## 注意事项

- 本仓库不包含原始数据集、训练权重和完整输出结果。
- `checkpoints/`、`ckpt/`、`outputs/`、`output/`、`datasets/` 等目录默认被 `.gitignore` 忽略。
- 高分实验使用了公开预训练权重初始化后的微调结果，不应表述为完全从零训练。
- 不同表格中的指标口径可能不同；论文和对比分析优先采用 `docs/所有模型指标汇总.md` 中统一复算后的保存图像口径。
