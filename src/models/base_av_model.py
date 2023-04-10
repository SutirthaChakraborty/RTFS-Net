import sys
import torch
import torch.nn as nn
import pytorch_lightning as ptl

from thop import profile


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

    def get_MACs(self):
        batch_size = 1
        seconds = 2

        audio_input = torch.rand(batch_size, seconds * 16000)
        if self.video_bn_params.get("is2d", False):
            video_input = torch.rand(batch_size, self.pretrained_vout_chan, seconds * 25, 16)
        else:
            video_input = torch.rand(batch_size, self.pretrained_vout_chan, seconds * 25)

        encoded_audio = self.encoder(audio_input)

        bn_audio = self.audio_bottleneck(encoded_audio)
        bn_video = self.video_bottleneck(video_input)

        separated_audio_embedding = self.mask_generator(bn_audio, encoded_audio)

        MACs = []

        MACs.append(int(profile(self.encoder, inputs=(audio_input,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.audio_bottleneck, inputs=(encoded_audio,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.audio_bottleneck.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.video_bottleneck, inputs=(video_input,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.video_bottleneck.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.refinement_module, inputs=(bn_audio, bn_video), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.refinement_module.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.mask_generator, inputs=(bn_audio, encoded_audio), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.mask_generator.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.decoder, inputs=(separated_audio_embedding, encoded_audio.shape), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.decoder.parameters() if p.requires_grad) / 1000))

        self.macs = int(profile(self, inputs=(audio_input, video_input), verbose=False)[0] / 1000000)

        self.trainable_params = int(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000)
        self.non_trainable_params = int(sum(p.numel() for p in self.parameters() if not p.requires_grad) / 1000)

        MACs.append(self.macs)
        MACs.append(self.trainable_params)
        MACs.append(self.non_trainable_params)

        MACs = ["{:,}".format(m) for m in MACs]

        s = (
            "CTCNet\n"
            "Encoder ----------- MACs: {:>8} M    Params: {:>6} K\n"
            "Audio BN ---------- MACs: {:>8} M    Params: {:>6} K\n"
            "Video BN ---------- MACs: {:>8} M    Params: {:>6} K\n"
            "RefinementModule -- MACs: {:>8} M    Params: {:>6} K\n"
            "Mask Generator ---- MACs: {:>8} M    Params: {:>6} K\n"
            "Decoder ----------- MACs: {:>8} M    Params: {:>6} K\n"
            "Total ------------- MACs: {:>8} M    Params: {:>6} K\n\n"
            "Non trainable params: {} K\n"
        ).format(*MACs)

        print(s)
