###
# Author: Kai Li
# Date: 2022-04-06 14:51:43
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 15:51:44
###
import warnings

warnings.filterwarnings("ignore")

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import *
from pytorch_lightning.loggers import TensorBoardLogger

from utils.datamodule import AVSpeechDataModule
from models.autoencoder import AE


def main():
    # dataloader
    datamodule = AVSpeechDataModule(
        "/home/likai/Autoencoder/code/LRS2/tr",
        "/home/likai/Autoencoder/code/LRS2/cv",
        "/home/likai/Autoencoder/code/LRS2/tt",
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
    checkpoint_dir = os.path.join("/home/likai/Autoencoder/exp")
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
    gpus = [0, 1, 2]
    distributed_backend = "gpu" if torch.cuda.is_available() else None

    # default logger used by trainer
    comet_logger = TensorBoardLogger(checkpoint_dir, name="baseline")

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        default_root_dir="/home/likai/Autoencoder/exp",
        devices=[1],
        accelerator=distributed_backend,
        strategy="ddp",
        limit_train_batches=1.0,  # Useful for fast experiment
        logger=comet_logger,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    print("Finished Training")


if __name__ == "__main__":
    main()
