###
# Author: Kai Li
# Date: 2022-04-06 14:51:43
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 15:51:44
###
import warnings

warnings.filterwarnings("ignore")

import os
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import *

from src.videomodels.autoencoder.datamodule import AVSpeechDataModule
from src.videomodels.autoencoder.autoencoder import AE

# ckpt_name = "epoch=199.ckpt"
ckpt_name = None


def main():
    # dataloader
    datamodule = AVSpeechDataModule(
        "data-preprocess/LRS2/tr",
        "data-preprocess/LRS2/cv",
        "data-preprocess/LRS2/tt",
        segment=2,
        batch_size=40,
    )
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader
    # Define scheduler
    system = AE(in_channels=1, base_channels=4, num_layers=3, train_loader=train_loader, val_loader=val_loader)

    # Define callbacks
    print("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join("../experiments/autoencoder", "default")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    callbacks.append(EarlyStopping(monitor="val/loss", patience=10, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    distributed_backend = "gpu" if torch.cuda.is_available() else None

    # default logger used by trainer
    comet_logger = TensorBoardLogger(checkpoint_dir, name="baseline")

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=False),
        limit_train_batches=1.0,  # Useful for fast experiment
        logger=comet_logger,
        # fast_dev_run=True,
    )
    trainer.fit(system, ckpt_path=os.path.join(checkpoint_dir, ckpt_name) if ckpt_name else None)
    print("Finished Training")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(checkpoint_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # put on cpu and serialize
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    torch.save(system.encoder.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
