###
# Author: Kai Li
# Date: 2022-04-03 08:50:42
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-03 18:02:56
###
###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-07 23:17:39
###
import os
import yaml
import torch
import warnings
import argparse
import numpy as np
import soundfile as sf

from torch.utils import data
from typing import OrderedDict

from src.models import CTCNet
from src.videomodels import FRCNNVideoModel
from src.utils.parser_utils import parse_args_as_dict
from src.losses import PITLossWrapper, pairwise_neg_sisdr
from src.datas.transform import get_preprocessing_pipelines

warnings.filterwarnings("ignore")


class EvalDataset(data.Dataset):
    def __init__(self, audio):
        self.audio_path = os.listdir(audio)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["val"]

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        spk, sr = sf.read(
            "/home/likai/data3/lrs2_asr/audio/{}".format(self.audio_path[index]),
            dtype="float32",
        )
        mixture = torch.from_numpy(spk)
        # mouth
        mp1 = self.audio_path[index].split("_")[0]
        mp2 = self.audio_path[index].split("_")[1]
        mouth_path = "/home/likai/data3/lrs2_asr/mouths/{}_{}.npz".format(mp1, mp2)
        # print(mouth_path)
        # import pdb; pdb.set_trace()
        s1_mouth = source_mouth = self.lipreading_preprocessing_func(np.load(mouth_path)["data"])
        s1_mouth = torch.from_numpy(s1_mouth)
        return mixture, s1_mouth, self.audio_path[index]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--conf_dir",
    default="/data/home/scv1134/run/chenhang/src-avdomain/egs/frcnn2_para/exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/conf.yml",
    help="Full path to save best validation model",
)

compute_metrics = ["si_sdr", "sdr"]


def load_ckpt(path, submodule=None):
    _state_dict = torch.load(path, map_location="cpu")["state_dict"]
    if submodule is None:
        return _state_dict

    state_dict = OrderedDict()
    for k, v in _state_dict.items():
        if submodule in k:
            L = len(submodule)
            state_dict[k[L + 1 :]] = v
    return state_dict


def main(conf):
    conf["exp_dir"] = os.path.join("exp", conf["log"]["exp_name"])
    conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "checkpoints/last.ckpt")
    videomodel = FRCNNVideoModel(**conf["videonet"])
    audiomodel = CTCNet(**conf["audionet"])
    ckpt = load_ckpt(model_path, "audio_model")
    audiomodel.load_state_dict(ckpt)

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    #     test_set = EvalDataset(
    #         "/home/likai/data3/lrs2_asr/audio"
    #     )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    for idx in range(1, 3):
        spk, sr = sf.read(
            "/data/home/scv1134/run/likai/av-separation-with-context/test_videos/interview/interview.wav",
            dtype="float32",
        )
        mouth = get_preprocessing_pipelines()["val"](
            np.load("/data/home/scv1134/run/likai/av-separation-with-context/test_videos/interview/mouthroi/speaker{}.npz".format(idx))[
                "data"
            ]
        )
        key = "spk{}".format(idx)

        # Forward the network on the mixture.
        target_mouths = torch.from_numpy(mouth).to(model_device)
        mix = torch.from_numpy(spk).to(model_device)
        # import pdb; pdb.set_trace()
        mouth_emb = videomodel(target_mouths.unsqueeze(0).unsqueeze(1).float())
        est_sources = audiomodel(mix[None, None], mouth_emb)

        gt_dir = "./test/sep_result"
        os.makedirs(gt_dir, exist_ok=True)
        # import pdb; pdb.set_trace()
        sf.write(
            os.path.join(gt_dir, key + ".wav"),
            est_sources.squeeze(0).squeeze(0).cpu().numpy(),
            16000,
        )
        # import pdb; pdb.set_trace()


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])
    main(def_conf)
