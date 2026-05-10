# Extra Benchmark Visuals

Additional thesis-ready grids for discussing ablations and cross-dataset behavior.

| File | Content |
|---|---|
| `rain100l_ablation_grid.png` | Rain100L samples comparing input, `zero_prompt`, `no_channel_attention`, MWIR-Net TTA, and target. |
| `rain100h_plain_tta_grid.png` | Rain100H samples comparing input, MWIR-Net plain inference, MWIR-Net TTA, and target. |
| `test1200_grid.png` | Test1200 deraining samples comparing input, MWIR-Net output, and target. |
| `test2800_grid.png` | Test2800 deraining samples comparing input, MWIR-Net output, and target. |
| `nyuhaze500_grid.png` | NYU-Haze500 dehazing samples comparing input, MWIR-Net output, and target. |

Generate these images with:

```bash
python tools/make_extra_thesis_visuals.py
```
