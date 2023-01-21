###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Kai Li
# LastEditTime: 2021-09-05 22:34:03
###
import os
import yaml
import torch
import argparse
import warnings

from tqdm import tqdm
from typing import OrderedDict

from src.utils import tensors_to_device
from src.metrics import ALLMetricsTracker
from src.videomodels import FRCNNVideoModel
from src.models.avfrcnn2 import AVFRCNN2CTC
from src.utils.parser_utils import parse_args_as_dict
from src.datas.avspeech_dataset import AVSpeechDataset
from src.losses import PITLossWrapper, pairwise_neg_sisdr


warnings.filterwarnings("ignore")


def main(conf):
    conf["exp_dir"] = os.path.join("../av-experiments", conf["log"]["exp_name"])
    # conf["audionet"].update({"n_src": 1})
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")

    sample_rate = conf["data"]["sample_rate"]
    audiomodel: torch.nn.Module = AVFRCNN2CTC.from_pretrain(model_path, sample_rate=sample_rate, **conf["audionet"])

    videomodel = FRCNNVideoModel(**conf["videonet"])

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    test_set = AVSpeechDataset(
        conf["test_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=None,
        normalize_audio=conf["data"]["normalize_audio"],
        return_src_path=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    metrics = ALLMetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()

    pbar = tqdm(range(len(test_set)))
    for idx in pbar:
        # Forward the network on the mixture.
        mix, sources, target_mouths, key, src_path = tensors_to_device(test_set[idx], device=model_device)
        mouth_emb = videomodel(target_mouths.unsqueeze(0).float())
        est_sources = audiomodel(mix[None, None], mouth_emb)
        loss, reordered_sources = loss_func(est_sources, sources[None, None], return_ests=True)
        mix_np = mix
        sources_np = sources[None]
        est_sources_np = reordered_sources.squeeze(0)
        metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)

        # print(sources_np.shape)
        # print(est_sources_np.shape)

        # save_dir = conf["save_dir"]
        # gt_dir = "vox_gt"
        # if save_dir is not None:
        #     # spk = src_path.split('/')[-2]
        #     # spk_id = ["s1", "s2"].index(spk)
        #     # splits = key.replace('.wav', '').split("/")[-1].split('_')
        #     # res = ["{}_{}_{}".format(splits[0], splits[1], splits[2]), "{}_{}_{}".format(splits[3], splits[4], splits[5])]
        #     # # import pdb; pdb.set_trace()
        #     # mouth_key = res[spk_id]
        #     spk = src_path.split('/')[-2]
        #     spk_id = ["s1", "s2"].index(spk)
        #     p = re.compile(r'id\d{5}_.{11}_\d{5}')
        #     res = p.findall(key)
        #     mouth_key = res[spk_id]

        #     if not osp.exists(save_dir):
        #         os.makedirs(save_dir)
        #     if not osp.exists(gt_dir):
        #         os.makedirs(gt_dir)
        #     est_sources_np = est_sources_np[0].cpu().numpy()
        #     sf.write(osp.join(save_dir, mouth_key+".wav"), est_sources_np, 16000)
        #     sources_np = sources_np[0].cpu().numpy()
        #     sf.write(osp.join(gt_dir, mouth_key+".wav"), sources_np, 16000)
        #     mix_np = mix_np.cpu().numpy()
        #     sf.write(osp.join(gt_dir, key), mix_np, 16000)

        if not (idx % 10):
            pbar.set_postfix(metrics.get_mean())

    metrics.final()
    mean, std = metrics.get_mean(), metrics.get_std()
    keys = list(mean.keys() & std.keys())

    order = ["sdr_i", "si-snr_i", "pesq", "stoi", "sdr", "si-snr"]

    def get_order(k):
        try:
            ind = order.index(k)
            return ind
        except ValueError:
            return 100

    keys.sort(key=get_order)
    for k in keys:
        m, s = mean[k], std[k]
        print(f"{k}\tmean: {m:.4f}  std: {s:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test-dir",
        type=str,
        default="data-preprocess/LRS2/tt",
        help="Test directory including the json files",
    )
    parser.add_argument(
        "-c",
        "--conf-dir",
        type=str,
        default="/home/anxihao/data2/av-experiments/ctcnet_pretrain_baseline_1_3/conf.yml",
        help="Full path to save best validation model",
    )
    parser.add_argument(
        "--n-save-ex",
        type=int,
        default=-1,
        help="Number of audio examples to save, -1 means all",
    )
    # parser.add_argument("-s", "--save-dir", default=None, help="Full path to save the results wav")

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])

    main(def_conf)
