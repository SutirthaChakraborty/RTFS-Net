###
# Author: Kai Li
# Date: 2021-06-20 01:32:22
# LastEditors: Kai Li
# LastEditTime: 2021-08-01 12:19:10
###
import comet_ml
import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.datas import AVSpeechDataset
from src.system import make_optimizer, System, CometLogger
from src.losses import SingleSrcNegSDR
from src.models import Wujian
from src.videomodels import WujianVideoModel
from src.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)

with open("local/lrs2_conf.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)

conf, plain_args = parse_args_as_dict(parser, return_plain_args=True)
conf["masknet"].update({"n_src": 1})

# model = Wujian(
#     **arg_dic["filterbank"], **arg_dic["masknet"], sample_rate=arg_dic["data"]["sample_rate"]
# )
# videonet = WujianVideoModel(**arg_dic["videonet"])
# a = torch.randn(3, 32000)
# v = torch.randn(3, 1, 100, 96, 96)
# v = videonet(v)
# print(v.shape)
# print(model(a, v).shape)

train_set = AVSpeechDataset(
    conf["data"]["train_dir"],
    n_src=1,
    sample_rate=conf["data"]["sample_rate"],
    segment=conf["data"]["segment"],
    normalize_audio=conf["data"]["normalize_audio"],
)

train_loader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=conf["training"]["batch_size"],
    num_workers=0,
    drop_last=True,
)

for batch in train_loader:
    mixture, sources, sources_mouths, _ = batch
    import pdb

    pdb.set_trace()
