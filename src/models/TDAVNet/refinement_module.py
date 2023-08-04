import inspect
import torch
import torch.nn as nn

from .fusion import MultiModalFusion
from .. import separators


class RefinementModule(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_params: dict,
    ):
        super(RefinementModule, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_params = fusion_params

        self.fusion_repeats = self.video_params.get("repeats", 0)
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats

        self.audio_net = separators.get(self.audio_params.get("audio_net", None))(
            **self.audio_params,
            in_chan=self.audio_bn_chan,
        )
        self.video_net = separators.get(self.video_params.get("video_net", None))(
            **self.video_params,
            in_chan=self.video_bn_chan,
        )

        self.crossmodal_fusion = MultiModalFusion(
            **self.fusion_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video

        # cross modal fusion
        for i in range(self.fusion_repeats):
            audio = self.audio_net.get_block(i)(audio + audio_residual if i > 0 else audio)
            video = self.video_net.get_block(i)(video + video_residual if i > 0 else video)

            audio, video = self.crossmodal_fusion.get_fusion_block(i)(audio, video)

        # further refinement
        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats

            audio = self.audio_net.get_block(i)(audio + audio_residual if i > 0 else audio)

        return audio

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

    def get_MACs(self, bn_audio, bn_video):
        from thop import profile

        audio_macs = int(profile(self.audio_net, inputs=(bn_audio,), verbose=False)[0] / 1000000)
        audio_params = int(sum(p.numel() for p in self.audio_net.parameters() if p.requires_grad) / 1000)

        video_macs = int(profile(self.video_net, inputs=(bn_video,), verbose=False)[0] / 1000000)
        video_params = int(sum(p.numel() for p in self.video_net.parameters() if p.requires_grad) / 1000)

        fusion_macs = int(profile(self.crossmodal_fusion, inputs=(bn_audio, bn_video), verbose=False)[0] / 1000000)
        fusion_params = int(sum(p.numel() for p in self.crossmodal_fusion.parameters() if p.requires_grad) / 1000)

        return audio_macs, audio_params, video_macs, video_params, fusion_macs, fusion_params
