import sys
import torch
import torch.nn as nn
import pytorch_lightning as ptl


class BaseAVModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_state_dict_in(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get

        conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")

        model_class = get(conf["model_name"])
        model = model_class(*args, **kwargs)
        model.load_state_dict(conf["state_dict"])

        return model

    def serialize(self):

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_config(),
        )

        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=ptl.__version__,
            python_version=sys.version,
        )

        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_config(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError
