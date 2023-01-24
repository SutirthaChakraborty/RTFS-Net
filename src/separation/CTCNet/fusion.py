import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ...layers import normalizations, FRCNNBlock
    from .frcnn import FRCNN as VideoFRCNN
except:
    from layers import normalizations, FRCNNBlock
    from frcnn import FRCNN as VideoFRCNN


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = 0,
        norm_type: str = "gLN",
    ):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, stride, padding, dilation, bias=True, groups=groups)
        self.norm = normalizations.get(norm_type)(out_chan)

    def forward(self, x):
        output = self.conv(x)
        return self.norm(output)


class Fusion_Base_module(nn.Module):
    def __init__(
        self,
        ain_chan: int = 128,
        vin_chan: int = 128,
    ):
        super(Fusion_Base_module, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan

    def forward(self, audio, video):
        raise NotImplementedError


class Concat_Fusion(Fusion_Base_module):
    def __init__(
        self,
        ain_chan: int = 128,
        vin_chan: int = 128,
    ):
        super(Concat_Fusion, self).__init__(ain_chan, vin_chan)
        self.audio_conv = ConvNorm(ain_chan + vin_chan, ain_chan, 1, 1)
        self.video_conv = ConvNorm(ain_chan + vin_chan, vin_chan, 1, 1)

    def forward(self, audio, video):
        video_interp = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio_video_concat = torch.cat([audio, video_interp], dim=1)
        audio_fused = self.audio_conv(audio_video_concat)

        audio_interp = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_audio_concat = torch.cat([audio_interp, video], dim=1)
        video_fused = self.video_conv(video_audio_concat)

        return audio_fused, video_fused


class Sum_Fusion(Fusion_Base_module):
    def __init__(self, ain_chan: int = 128, vin_chan: int = 128):
        super(Sum_Fusion, self).__init__(ain_chan, vin_chan)
        self.audio_conv = ConvNorm(vin_chan, ain_chan, 1, 1)
        self.video_conv = ConvNorm(ain_chan, vin_chan, 1, 1)

    def forward(self, audio, video):
        audio_interp = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_fused = self.video_conv(audio_interp) + video

        video_interp = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio_fused = self.audio_conv(video_interp) + audio

        return audio_fused, video_fused


class Multi_Modal_Fusion(nn.Module):
    def __init__(
        self,
        audio_bn_chan: int,
        video_bn_chan: int,
        audio_frcnn: FRCNNBlock,
        video_frcnn: VideoFRCNN,
        fusion_repeats: int = 3,
        audio_repeats: int = 3,
        fusion_type: str = "Concat_Fusion",
        fusion_shared: bool = False,
    ):
        super(Multi_Modal_Fusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.audio_frcnn = audio_frcnn
        self.video_frcnn = video_frcnn
        self.fusion_repeats = fusion_repeats
        self.audio_repeats = audio_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared

        self.audio_concat = nn.Sequential(nn.Conv1d(self.audio_bn_chan, self.audio_bn_chan, 1, groups=self.audio_bn_chan), nn.PReLU())

        fusion_class: Fusion_Base_module = globals()[self.fusion_type]

        if self.fusion_shared:
            self.fusion_module = fusion_class(self.audio_bn_chan, self.video_bn_chan)
        else:
            self.fusion_module = nn.ModuleList([fusion_class(self.audio_bn_chan, self.video_bn_chan) for _ in range(self.fusion_repeats)])

    def __get_crossmodal_fusion(self, i):
        if self.fusion_shared:
            return self.fusion_module
        else:
            return self.fusion_module[i]

    def forward(self, audio, video):
        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio = self.audio_frcnn(audio)
                video = self.video_frcnn.get_frcnn_block(i)(video)
                audio_fused, video_fused = self.__get_crossmodal_fusion(i)(audio, video)
            else:
                audio_fused = self.audio_frcnn(self.audio_concat(audio_fused + audio_residual))
                video_fused = self.video_frcnn.get_frcnn_block(i)(self.video_frcnn.get_concat_block(i)(video_fused + video_residual))
                audio_fused, video_fused = self.__get_crossmodal_fusion(i)(audio_fused, video_fused)

        for i in range(self.audio_repeats):
            audio_fused = self.audio_frcnn(self.audio_concat(audio_fused + audio_residual))

        return audio_fused
