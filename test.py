import os
import sys
import yaml
import argparse
import warnings
import importlib
import torchaudio
import pandas as pd
from sigfig import round
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.metrics import ALLMetricsTracker
from src.utils.parser_utils import parse_args_as_dict
from src.datas.avspeech_dataset import AVSpeechDataset
from src.losses import PITLossWrapper, pairwise_neg_sisdr
from src.utils import tensors_to_device

warnings.filterwarnings("ignore")


class TestModule(pl.LightningModule):
    def __init__(self, conf):
        super(TestModule, self).__init__()
        self.conf = conf
        self.conf["videonet"] = conf.get("videonet", {})
        self.conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)

        exp_dir = os.path.abspath(os.path.join("../experiments/audio-visual", conf["log"]["exp_name"]))
        sys.path.append(os.path.dirname(exp_dir))
        models_module = importlib.import_module(os.path.basename(exp_dir) + ".models")
        videomodels = importlib.import_module(os.path.basename(exp_dir) + ".models.videomodels")
        AVNet = getattr(models_module, "AVNet")

        model_path = os.path.join(self.conf["exp_dir"], "best_model.pth")
        self.audiomodel = AVNet.from_pretrain(model_path, **self.conf["audionet"])
        self.videomodel = None
        if self.conf["videonet"]["model_name"]:
            self.videomodel = videomodels.get(self.conf["videonet"]["model_name"])(**self.conf["videonet"])

        self.loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.metrics = ALLMetricsTracker(save_file=os.path.join(self.conf["exp_dir"], "results", "metrics.csv"))

    def test_step(self, batch, batch_idx):
        mix, sources, target_mouths, key, src_path = tensors_to_device(batch, device=self.device)
        mouth_emb = self.videomodel(target_mouths.unsqueeze(0).float()) if self.videomodel is not None else None
        est_sources = self.audiomodel(mix[None, None], mouth_emb)
        loss, reordered_sources = self.loss_func(est_sources, sources[None, None], return_ests=True)
        self.log("test_loss", loss)
        return {"mix": mix, "sources": sources, "est_sources": reordered_sources, "key": key}

    def test_epoch_end(self, outputs):
        ex_save_dir = os.path.join(self.conf["exp_dir"], "results/")
        os.makedirs(ex_save_dir, exist_ok=True)
        for idx, output in enumerate(outputs):
            mix_np, sources_np, est_sources_np, key = output["mix"], output["sources"], output["est_sources"], output["key"]
            self.metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)

            if idx < self.conf["n_save_ex"]:
                if not os.path.exists(os.path.join(ex_save_dir, "examples")):
                    os.makedirs(os.path.join(ex_save_dir, "examples"))

                est_sources_np = est_sources_np[0].cpu().unsqueeze(0)
                torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_est.wav"), est_sources_np, 16000)
                sources_np = sources_np[0].cpu().unsqueeze(0)
                torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_gt.wav"), sources_np, 16000)
                mix_np = mix_np.cpu().unsqueeze(0)
                torchaudio.save(os.path.join(ex_save_dir, "examples", str(idx) + "_mix.wav"), mix_np, 16000)

        self.metrics.final()
        mean, std = self.metrics.get_mean(), self.metrics.get_std()
        keys = list(mean.keys() & std.keys())

        order = ["si-snr_i", "sdr_i", "pesq", "stoi", "si-snr", "sdr"]

        def get_order(k):
            try:
                ind = order.index(k)
                return ind
            except ValueError:
                return 100

        results_dict = []

        results_dict.append(("Model", self.conf["log"]["exp_name"]))
        results_dict.append(("MACs and Params", self.audiomodel.macs_parms))
        results_dict.append(("Videomodel MACs", self.videomodel.macs))
        results_dict.append(("Videomodel Params", self.videomodel.trainable_params))

        keys.sort(key=get_order)
        for k in keys:
            m, s = round(mean[k], 4), round(std[k], 3)
            results_dict.append((k, str(m) + " ± " + str(s)))
            print(f"{k}\tmean: {m}  std: {s}")

        for k, v in self.conf["audionet"].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    results_dict.append((k + "_" + kk, vv))
            else:
                results_dict.append((k, v))

        df = pd.DataFrame.from_records(results_dict, columns=["Key", "Value"])
        df.to_csv(os.path.join(ex_save_dir, "results.csv"), encoding="utf-8", index=False)

    def test_dataloader(self):
        test_set = AVSpeechDataset(
            self.conf["test_dir"],
            n_src=self.conf["data"]["nondefault_nsrc"],
            sample_rate=self.conf["data"]["sample_rate"],
            segment=None,
            normalize_audio=self.conf["data"]["normalize_audio"],
            return_src_path=True,
        )
        return DataLoader(test_set, batch_size=1, shuffle=False)


def main(conf):
    model = TestModule(conf)
    trainer = pl.Trainer(
        default_root_dir=conf["log"]["exp_dir"],
        devices=conf["training"]["gpus"],
        num_nodes=conf["main_args"]["nodes"],
        accelerator="auto",
    )
    trainer.test(model)


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
        default="../experiments/audio-visual/",
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
