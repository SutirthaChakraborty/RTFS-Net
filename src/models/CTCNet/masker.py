import inspect
import torch.nn as nn

from .fusion import MultiModalFusion
from ...models import layers


class RefinementModule(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_type: str,
        fusion_shared: bool,
    ):
        super(RefinementModule, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared

        self.fusion_repeats = self.video_params["repeats"]
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats

        self.video_net = layers.get(self.video_params["video_net"])(**self.video_params, in_chan=self.video_bn_chan)
        self.audio_net = layers.get(self.audio_params["audio_net"])(**self.audio_params, in_chan=self.audio_bn_chan)

        self.crossmodal_fusion = MultiModalFusion(
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
            audio_repeats=self.audio_repeats,
            fusion_type=self.fusion_type,
            fusion_shared=self.fusion_shared,
        )

    def forward(self, audio, video):

        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio = self.audio_net.get_block(i)(audio)
                video = self.video_net.get_block(i)(video)
                audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio, video)
            else:
                audio_fused = self.audio_net.get_block(i)(self.audio_net.get_concat_block(i)(audio_fused + audio_residual))
                video_fused = self.video_net.get_block(i)(self.video_net.get_concat_block(i)(video_fused + video_residual))
                audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio_fused, video_fused)

        for i in range(self.audio_repeats):
            j = i + self.fusion_repeats
            audio_fused = self.audio_net.get_block(j)(self.audio_net.get_concat_block(j)(audio_fused + audio_residual))

        return audio_fused

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
