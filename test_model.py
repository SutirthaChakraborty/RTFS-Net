###
# Author: Kai Li
# Date: 2021-06-20 01:32:22
# LastEditors: Kai Li
# LastEditTime: 2021-08-01 11:42:27
###
# import comet_ml
import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.datas.audio_dataset import AudioDataset
from src.system.optimizers import make_optimizer
from src.system.core import System

# from src.system.comet import CometLogger
from src.losses import PITLossWrapper, pairwise_neg_sisdr
from src.models.avfrcnn import AVFRCNN
from src.videomodels import FRCNNVideoModel, update_frcnn_parameter
from src.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
from src.models.avfrcnn2 import AVFRCNN2
from src.models.avfrcnn2 import AVFRCNN2CTC
from src.models.avfrcnn2 import AVFRCNN2CTCCC
from src.models.avfrcnn2 import AVFRCNN2FPN
from src.models.avfrcnn2 import AVFRCNN2TOP
from src.models.avfrcnn2 import AVFRCNN2PARA
import yaml

# from thop import clever_format
# from thop import profile


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)

with open("local/lrs2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)

arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
sample_rate = arg_dic["data"]["sample_rate"]
videomodel = FRCNNVideoModel(**arg_dic["videonet"])
audiomodel = AVFRCNN2(sample_rate=sample_rate, **arg_dic["audionet"])
# print(v1 == v2)
a = torch.randn(1, 32000)
# v = torch.randn(3, 100, 96, 96)
v = torch.randn(1, 1, 100, 96, 96)
v = videomodel(v)
print(v.shape)
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    net = audiomodel
    macs, params = get_model_complexity_info(
        net, (a, v), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
print(macs)
print(params)
print(check_parameters(audiomodel) + check_parameters(videomodel))
