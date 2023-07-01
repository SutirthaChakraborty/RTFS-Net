import os
import yaml
import torch
import argparse

torch.set_float32_matmul_precision("high")

from src.models import TDAVNet
from src.utils import parse_args_as_dict
from src.system import make_optimizer
from src.losses import PITLossWrapper, pairwise_neg_snr


x = torch.rand(2, 32000)
z = torch.rand(2, 1, 32000)
y = torch.rand(2, 121, 50, 16)


def main(conf):
    audiomodel = TDAVNet(**conf["audionet"])

    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    optimizer.zero_grad()

    z1 = audiomodel(x, y)

    loss = loss_func(z1, z)
    loss.backward()

    for name, param in audiomodel.named_parameters():
        if param.grad is None:
            print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf-dir", default="config/lrs2_tdanet2d_small.yml")
    parser.add_argument("-n", "--name", default=None, help="Experiment name")
    parser.add_argument("--nodes", type=int, default=1, help="#node")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint if training crashes")

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf["log"]["exp_name"] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)
