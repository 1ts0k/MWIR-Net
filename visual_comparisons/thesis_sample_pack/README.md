# Thesis Sample Pack

This folder keeps a compact set of paper-ready original pairs and model outputs.
It mirrors the figures referenced in the thesis without copying the full output archive.
Each group follows `input/`, `target/`, `outputs/<model>/`, and `preview/`.
Preview collages use the same left-to-right column order listed in the table below.

| Group | Samples | Columns |
|---|---|---|
| `rain100l_main` | `1, 10, 25, 50, 100` | Input, Restormer-official, PromptIR-official, MPRNet-official, AirNet-official-All, PromptIR-5k-init, AirNet-5k, MWIR-Net-5k-init, MWIR-Net-TTA, Target |
| `rain100l_ablation` | `1, 25, 100` | Input, Zero Prompt, No Channel Attention, Target |
| `sots_outdoor` | `0001_0.8_0.2, 0030_0.95_0.12, 0056_0.8_0.16, 0066_1_0.08, 0100_0.9_0.12` | Input, PromptIR-official, AirNet-official-All, PromptIR-5k-init, AirNet-5k, MWIR-Net-5k-init, MWIR-Net-TTA, Target |
| `rain100h` | `1, 25, 100` | Input, MWIR-Net Plain, MWIR-Net TTA, Target |
| `test1200` | `1, 600, 1200` | Input, MWIR-Net, Target |
| `test2800` | `801_1, 802_1, 803_1` | Input, MWIR-Net, Target |
| `nyuhaze500` | `1400_1, 1400_6, 1401_1` | Input, MWIR-Net, Target |

Regenerate the pack with:

```bash
python tools/export_thesis_sample_pack.py
```

The `manifest.csv` file records every copied image and preview path.
