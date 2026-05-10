# Original Samples

Curated raw input/target pairs used by the thesis figures.

| Folder | Contents |
|---|---|
| `rain100l/` | Rain100L input and target pairs for samples `1`, `10`, `25`, `50`, `100`. |
| `rain100h/` | Rain100H input and target pairs for samples `1`, `25`, `100`. |
| `test1200/` | Test1200 input and target pairs for samples `1`, `600`, `1200`. |
| `test2800/` | Test2800 input and target pairs for samples `801_1`, `802_1`, `803_1`. |
| `sots_outdoor/` | SOTS outdoor input and target pairs for samples `0001`, `0030`, `0056`, `0066`, `0100`. |
| `nyuhaze500/` | NYU-Haze500 input and target pairs for samples `1400_1`, `1400_6`, `1401_1`. |

Generate or refresh them with:

```bash
python tools/export_original_samples.py
```
