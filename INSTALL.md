# Install

The current machine already has the required Python environment. If you need to recreate it, use:

```bash
conda env create -f env.yml
conda activate mwirnet
```

Prepare data links after the datasets are placed in `datasets/`:

```bash
python tools/prepare_mwir_data.py --dehaze-source ots
```

Run a 1-step smoke test:

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
