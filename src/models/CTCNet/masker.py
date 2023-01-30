import torch
import inspect
import torch.nn as nn

from .fusion import MultiModalFusion
from ..layers import normalizations, activations, FRCNN


class Masker(nn.Module):
    def __init__(
        self,
        n_src: int,
        in_chan: int,
        hid_chan: int,
        audio_bn_chan: int,
        fusion_repeats: int = 1,
        audio_repeats: int = 3,
        upsampling_depth: int = 5,
        norm_type: str = "gLN",
        mask_act: str = "ReLU",
        act_type: str = "PReLU",
        video_frcnn: dict = dict(),
        pretrain: str = None,
        audio_shared: bool = True,
        fusion_shared: bool = False,
        fusion_type: str = "ConcatFusion",
    ):
        super(Masker, self).__init__()
        self.n_src = n_src
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_frcnn["in_chan"]
        self.vout_chan = video_frcnn["vout_chan"]
        self.fusion_repeats = fusion_repeats
        self.audio_repeats = audio_repeats
        self.upsampling_depth = upsampling_depth
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.act_type = act_type
        self.pretrain = pretrain
        self.audio_shared = audio_shared
        self.fusion_shared = fusion_shared
        self.fusion_type = fusion_type

        self.audio_bottleneck = nn.Sequential(
            normalizations.get(self.norm_type)(self.in_chan),
            nn.Conv1d(self.in_chan, self.audio_bn_chan, 1, 1),
        )
        self.video_bottleneck = nn.Conv1d(self.vout_chan, self.video_bn_chan, kernel_size=3, padding=1)
        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.audio_bn_chan, self.n_src * self.in_chan, 1, 1),
            activations.get(self.mask_act)(),
        )

        # main modules
        self.video_frcnn = FRCNN(**video_frcnn)
        self.audio_frcnn = FRCNN(
            in_chan=self.audio_bn_chan,
            out_chan=self.hid_chan,
            upsampling_depth=self.upsampling_depth,
            repeats=self.audio_repeats + self.fusion_repeats,
            shared=self.audio_shared,
            norm_type=self.norm_type,
            act_type=self.act_type,
        )

        self.crossmodal_fusion = MultiModalFusion(
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats,
            audio_repeats=self.audio_repeats,
            fusion_type=self.fusion_type,
            fusion_shared=self.fusion_shared,
        )

        if self.pretrain is not None:
            state_dict = torch.load(self.pretrain, map_location="cpu")["model_state_dict"]

            frcnn_state_dict = dict()
            for k, v in state_dict.items():
                if k.startswith("module.head.frcnn"):
                    frcnn_state_dict[k[18:]] = v
            self.video_frcnn.load_state_dict(frcnn_state_dict)

            pre_v_state_dict = dict(weight=state_dict["module.head.proj.weight"], bias=state_dict["module.head.proj.bias"])
            self.video_bottleneck.load_state_dict(pre_v_state_dict)

    def forward(self, audio, video):
        batch_size = audio.shape[0]
        
        audio = self.audio_bottleneck(audio)
        video = self.video_bottleneck(video)

        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio = self.audio_frcnn.get_frcnn_block(i)(audio)
                video = self.video_frcnn.get_frcnn_block(i)(video)
                audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio, video)
            else:
                audio_fused = self.audio_frcnn.get_frcnn_block(i)(self.audio_frcnn.get_concat_block(i)(audio_fused + audio_residual))
                video_fused = self.video_frcnn.get_frcnn_block(i)(self.video_frcnn.get_concat_block(i)(video_fused + video_residual))
                audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio_fused, video_fused)

        for i in range(self.audio_repeats):
            j = i + self.fusion_repeats
            audio_fused = self.audio_frcnn.get_frcnn_block(j)(self.audio_frcnn.get_concat_block(j)(audio_fused + audio_residual))

        masks = self.mask_generator(audio_fused).view(batch_size, self.n_src, self.in_chan, -1)

        return masks

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
