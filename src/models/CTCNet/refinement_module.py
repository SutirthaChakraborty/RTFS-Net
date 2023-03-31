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

        self.fusion_repeats = self.video_params["repeats"]
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats

        self.video_net = separators.get(self.video_params["video_net"])(**self.video_params, in_chan=self.video_bn_chan)
        self.audio_net = separators.get(self.audio_params["audio_net"])(**self.audio_params, in_chan=self.audio_bn_chan)

        self.crossmodal_fusion = MultiModalFusion(
            **self.fusion_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
            audio_repeats=self.audio_repeats,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor = None):
        audio_residual = audio
        video_residual = video

        # cross modal fusion
        for i in range(self.fusion_repeats):
            if i > 0:
                audio = audio + audio_residual
                video = video + video_residual if video is not None else video

            audio = self.audio_net.get_block(i)(self.audio_net.get_concat_block(i)(audio))
            video = self.video_net.get_block(i)(self.video_net.get_concat_block(i)(video)) if video is not None else video
            audio, video = self.crossmodal_fusion.get_fusion_block(i)(audio, video) if video is not None else audio, video

        # further refinement
        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats

            audio = self.audio_net.get_concat_block(i)(audio + audio_residual)
            audio = self.audio_net.get_block(i)(audio)

        return audio

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
