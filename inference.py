import os
import sys
import torch
import importlib
import torchaudio
from sigfig import round
import yaml
from src.metrics import ALLMetricsTracker
from src.utils.parser_utils import parse_args_as_dict
from src.losses import PITLossWrapper, pairwise_neg_sisdr
from src.datas.transform import get_preprocessing_pipelines
import numpy as np
import soundfile as sf
import argparse

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class TestOneVideo():
    def __init__(self, conf):
        super(TestOneVideo, self).__init__()
        self.conf = conf
        self.conf["videonet"] = conf.get("videonet", {})
        self.conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)
        self.exp_dir = os.path.abspath(os.path.join("../experiments/audio-visual", conf["log"]["exp_name"]))

        sys.path.append(os.path.dirname(self.exp_dir))
        models_module = importlib.import_module(os.path.basename(self.exp_dir) + ".models")
        videomodels = importlib.import_module(os.path.basename(self.exp_dir) + ".models.videomodels")
        AVNet = getattr(models_module, "AVNet")

        model_path = os.path.join(self.exp_dir, "best_model.pth")
        self.audiomodel = AVNet.from_pretrain(model_path, **self.conf["audionet"])
        self.videomodel = None
        if self.conf["videonet"]["model_name"]:
            self.videomodel = videomodels.get(self.conf["videonet"]["model_name"])(**self.conf["videonet"], print_macs=False)

    def test(self):
        with torch.no_grad():
            for idx in range(1, 2):
                file_name='/data/schakraborty/jusper/lrs2_rebuild/audio/wav16k/min/tt/mix/6330311066473698535_00011_0.53084_6339356267468615354_00010_-0.53084.wav'
                mix, fs = sf.read(file_name, dtype="float32")
                mixture=torch.from_numpy(mix)
                m_std = mixture.std(-1, keepdim=True)
                EPS = 1e-8
                mixture = normalize_tensor_wav(mixture, eps=EPS, std=m_std)[: int(16000 * 2)].squeeze(0)
                mouth_path='/data/schakraborty/jusper/lrs2_rebuild/mouths/6330311066473698535_00011.npz'
                target_mouths = get_preprocessing_pipelines()["val"](np.load(mouth_path)["data"])
                target_mouths=torch.from_numpy(np.load(mouth_path)["data"]).unsqueeze(0)
                mouth_emb = self.videomodel(target_mouths.float().unsqueeze(0)) if self.videomodel is not None else None
                est_sources = self.audiomodel(mixture.unsqueeze(0), mouth_emb)
                est_sources_np = est_sources.cpu().squeeze(0)
                torchaudio.save('infer_pred.wav', est_sources_np, 16000)
                mix_np = mixture.cpu().unsqueeze(0)
                torchaudio.save(os.path.join("infer_mix.wav"), mix_np, 16000)
               


def main(conf):
    model = TestOneVideo(conf)
    model.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf-dir",
        type=str,
         default="../experiments/audio-visual/RTFS-Net/LRS2/12_layers/conf.yaml",
        help="Full path to save best validation model",
    )

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])

    main(def_conf)
