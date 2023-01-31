###
# Author: Kai Li
# Date: 2021-06-20 00:21:33
# LastEditors: Kai Li
# LastEditTime: 2021-09-09 23:12:28
###

import os
import yaml
import json
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.system.core import System
from src.datas import AVSpeechDataset
from src.models import CTCNet
from src.videomodels import FRCNNVideoModel
from src.system.optimizers import make_optimizer
from src.utils.parser_utils import parse_args_as_dict
from src.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr


def build_dataloaders(conf):
    train_set = AVSpeechDataset(
        conf["data"]["train_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        normalize_audio=conf["data"]["normalize_audio"],
    )
    val_set = AVSpeechDataset(
        conf["data"]["valid_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=None,
        normalize_audio=conf["data"]["normalize_audio"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader


def main(conf):

    train_loader, val_loader = build_dataloaders(conf)

    # Define model and optimizer
    videomodel = FRCNNVideoModel(**conf["videonet"])
    audiomodel = CTCNet(**conf["audionet"])
    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)

    # Just after instantiating, save the args. Easy loading in the future.
    conf["main_args"]["exp_dir"] = os.path.join("../experiments/audio-visual", conf["log"]["exp_name"])
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = {
        "train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx"),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
    }

    # define system
    system = System(
        audio_model=audiomodel,
        video_model=videomodel,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=15, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = conf["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "gpu" if torch.cuda.is_available() else None

    # default logger used by trainer
    comet_logger = TensorBoardLogger("./logs", name=conf["log"]["exp_name"])

    # instantiate ptl trainer
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        num_nodes=conf["main_args"]["nodes"],
        accelerator=distributed_backend,
        strategy=DDPStrategy(),
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
    )

    trainer.fit(system)

    # Save best_k models
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # put on cpu and serialize
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    # Keys which are not in the conf.yml file can be added here.
    # In the hierarchical dictionary created when parsing, the key `key` can be
    # found at dic['main_args'][key]

    # By default train.py will use all available GPUs. The `id` option in run.sh
    # will limit the number of available GPUs for train.py .
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf-dir", default="config/lrs2_conf_small_fcrnn.yml")
    parser.add_argument("-n", "--name", default=None, help="Experiment name")
    parser.add_argument("--nodes", type=int, default=1, help="#node")

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf["log"]["exp_name"] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)
