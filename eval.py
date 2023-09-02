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
import torchaudio
import pandas as pd

from tqdm import tqdm
from sigfig import round

from src.models import TDAVNet
from src.utils import tensors_to_device, get_free_gpu_indices
from src.metrics import ALLMetricsTracker
from src.utils.parser_utils import parse_args_as_dict
from src.datas.avspeech_dataset import AVSpeechDataset
from src.losses import PITLossWrapper, pairwise_neg_sisdr
from src.videomodels import FRCNNVideoModel, AEVideoModel


warnings.filterwarnings("ignore")


def main(conf):
    conf["exp_dir"] = os.path.join("../experiments/audio-visual", conf["log"]["exp_name"])
    conf["videonet"] = conf.get("videonet", {})
    conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)

    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    audiomodel: TDAVNet = TDAVNet.from_pretrain(model_path, **conf["audionet"])
    audiomodel.get_MACs()
    videomodel = None
    if conf["videonet"]["model_name"] == "FRCNNVideoModel":
        videomodel = FRCNNVideoModel(**conf["videonet"])
    elif conf["videonet"]["model_name"] == "EncoderAE":
        videomodel = AEVideoModel(**conf["videonet"])

    # Handle device placement
    audiomodel.eval()
    if videomodel is not None:
        videomodel.eval()

    device = get_free_gpu_indices()
    if len(device):
        device = device[0]
        audiomodel.cuda(device)
        if videomodel is not None:
            videomodel.cuda(device)
    else:
        print("No free gpus available, using CPU")
        audiomodel.cpu()
        if videomodel is not None:
            videomodel.cpu()

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
    metrics = ALLMetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()

    pbar = tqdm(range(len(test_set)))
    for idx in pbar:
        # Forward the network on the mixture.
        mix, sources, target_mouths, key, src_path = tensors_to_device(test_set[idx], device=model_device)
        mouth_emb = videomodel(target_mouths.unsqueeze(0).float()) if videomodel is not None else None
        est_sources = audiomodel(mix[None, None], mouth_emb)
        loss, reordered_sources = loss_func(est_sources, sources[None, None], return_ests=True)
        mix_np = mix
        sources_np = sources[None]
        est_sources_np = reordered_sources.squeeze(0)
        metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)

        if idx < conf["n_save_ex"]:
            if not os.path.exists(os.path.join(ex_save_dir, "examples")):
                os.makedirs(os.path.join(ex_save_dir, "examples"))

            est_sources_np = est_sources_np[0].cpu().unsqueeze(0)
            torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_est.wav"), est_sources_np, 16000)
            sources_np = sources_np[0].cpu().unsqueeze(0)
            torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_gt.wav"), sources_np, 16000)
            mix_np = mix_np.cpu().unsqueeze(0)
            torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_mix.wav"), mix_np, 16000)

        if not (idx % 10):
            pbar.set_postfix(metrics.get_mean())

    metrics.final()
    mean, std = metrics.get_mean(), metrics.get_std()
    keys = list(mean.keys() & std.keys())

    order = ["si-snr_i", "sdr_i", "pesq", "stoi", "si-snr", "sdr"]

    def get_order(k):
        try:
            ind = order.index(k)
            return ind
        except ValueError:
            return 100

    results_dict = []

    results_dict.append(("Model", conf["log"]["exp_name"]))
    results_dict.append(("CTCNet MACs and Params", audiomodel.macs_parms))
    results_dict.append(("Videomodel MACs", videomodel.macs))
    results_dict.append(("Videomodel Params", videomodel.trainable_params))

    keys.sort(key=get_order)
    for k in keys:
        m, s = round(mean[k], 4), round(std[k], 3)
        results_dict.append((k, str(m) + " ± " + str(s)))
        print(f"{k}\tmean: {m}  std: {s}")

    for k, v in conf["audionet"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                results_dict.append((k + "_" + kk, vv))
        else:
            results_dict.append((k, v))

    df = pd.DataFrame.from_records(results_dict, columns=["Key", "Value"])
    df.to_csv(os.path.join(ex_save_dir, "results.csv"), encoding="utf-8", index=False)


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
        default="/ssd2/anxihao/experiments/audio-visual/",
        help="Full path to save best validation model",
    )
    parser.add_argument(
        "--n-save-ex",
        type=int,
        default=-1,
        help="Number of audio examples to save, -1 means none",
    )

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])

    main(def_conf)
