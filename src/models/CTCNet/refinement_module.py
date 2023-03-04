import inspect
import torch
import torch.nn as nn

from .fusion import MultiModalFusion
from .. import layers


class RefinementModule(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_type: str,
        fusion_shared: bool,
        gc3_params: dict = dict(),
    ):
        super(RefinementModule, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared
        self.gc3_params = gc3_params

        self.fusion_repeats = self.video_params["repeats"]
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats

        self.video_net = layers.get(self.video_params["video_net"])(
            **self.video_params,
            in_chan=self.video_bn_chan,
            group_size=self.gc3_params.get("video", dict()).get("group_size", 1),
        )
        self.audio_net = layers.get(self.audio_params["audio_net"])(
            **self.audio_params,
            in_chan=self.audio_bn_chan,
            group_size=self.gc3_params.get("audio", dict()).get("group_size", 1),
        )

        self.crossmodal_fusion = MultiModalFusion(
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
            audio_repeats=self.audio_repeats,
            fusion_type=self.fusion_type,
            fusion_shared=self.fusion_shared,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        batch_size, _, T1 = audio.shape
        T2 = video.shape[-1]

        audio_residual = audio
        video_residual = video

        audio_fused = audio
        video_fused = video

        for i in range(self.fusion_repeats):
            if i > 0:
                audio_fused = audio_fused + audio_residual
                video_fused = video_fused + video_residual

            audio_fused = self.audio_net.get_tac(i)(audio_fused.view(batch_size, self.audio_net.group_size, -1, T1))
            audio_fused = audio_fused.view(batch_size * self.audio_net.group_size, -1, T1)

            video_fused = self.video_net.get_tac(i)(video_fused.view(batch_size, self.video_net.group_size, -1, T2))
            video_fused = video_fused.view(batch_size * self.video_net.group_size, -1, T2)

            audio_fused = self.audio_net.get_block(i)(self.audio_net.get_concat_block(i)(audio_fused)).view(batch_size, -1, T1)
            video_fused = self.video_net.get_block(i)(self.video_net.get_concat_block(i)(video_fused)).view(batch_size, -1, T2)
            audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio_fused, video_fused)

        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats
            audio_fused = audio_fused + audio_residual

            audio_fused = self.audio_net.get_tac(i)(audio_fused.view(batch_size, self.audio_net.group_size, -1, T1))
            audio_fused = audio_fused.view(batch_size * self.audio_net.group_size, -1, T1)
            audio_fused = self.audio_net.get_block(i)(self.audio_net.get_concat_block(i)(audio_fused)).view(batch_size, -1, T1)

        return audio_fused

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
