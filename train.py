import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from net.mwirnet import MWIRNet
from options import options as opt
from utils.dataset_utils import MWIRTrainDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def gradient(self, image):
        gray = image.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        return grad_x, grad_y

    def forward(self, restored, target):
        restored_x, restored_y = self.gradient(restored)
        target_x, target_y = self.gradient(target)
        return F.l1_loss(restored_x, target_x) + F.l1_loss(restored_y, target_y)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, restored, target):
        diff = restored - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class MWIRLitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = MWIRNet(decoder=True, ablation_mode=opt.ablation_mode)
        pixel_loss_type = str(getattr(opt, "pixel_loss_type", "l1")).lower()
        if pixel_loss_type == "charbonnier":
            self.pixel_loss = CharbonnierLoss(eps=float(getattr(opt, "charbonnier_eps", 1e-3)))
        else:
            self.pixel_loss = nn.L1Loss()
        self.edge_loss = SobelEdgeLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        (_, degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        pixel_loss = self.pixel_loss(restored, clean_patch)
        edge_weight = float(getattr(opt, "edge_loss_weight", 0.0))
        edge_loss = self.edge_loss(restored, clean_patch) if edge_weight > 0 else restored.new_tensor(0.0)
        loss = pixel_loss + edge_weight * edge_loss

        self.log("train_loss", loss)
        self.log("pixel_loss", pixel_loss)
        self.log("edge_loss", edge_loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        max_epochs = max(1, int(opt.epochs))
        warmup_epochs = min(max(1, int(getattr(opt, "warmup_epochs", 2))), max_epochs)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        )
        return [optimizer], [scheduler]


def load_compatible_init(model, ckpt_path):
    if ckpt_path is None or str(ckpt_path).lower() in {"", "none", "false", "off"}:
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(compatible_state, strict=False)
    print(
        "Initialized MWIR-Net from {}: loaded {} tensors, skipped {} tensors.".format(
            ckpt_path, len(compatible_state), len(skipped)
        )
    )


def main():
    print("Options")
    print(opt)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    use_wandb = opt.wblogger is not None and str(opt.wblogger).lower() not in {"", "0", "none", "false", "off"}
    if use_wandb:
        logger = WandbLogger(project=opt.wblogger, name="MWIRNet-Train")
    else:
        logger = CSVLogger(save_dir="logs", name="mwirnet")

    trainset = MWIRTrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1,
    )

    if torch.cuda.is_available():
        devices = min(max(1, opt.num_gpus), torch.cuda.device_count())
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"
    model = MWIRLitModel()
    load_compatible_init(model, opt.init_ckpt)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        max_steps=opt.max_steps,
        precision=opt.precision,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
