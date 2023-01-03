###
# Author: Kai Li
# Date: 2021-06-21 15:21:58
# LastEditors: Kai Li
# LastEditTime: 2021-09-02 21:25:44
###

from typing import Iterable, Union
import torch
import math
import warnings
import inspect
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..base_av_model import BaseAVEncoderMaskerDecoder
from ...layers import (
    normalizations,
    activations,
    Conv1DBlock,
    ConvNormAct,
    ConvNorm,
    Video1DConv,
    Concat,
    FRCNNBlock,
    make_enc_dec,
    MultiHeadedSelfAttentionModule
)
from .frcnn import FRCNN as VideoFRCNN
from .modules.transformer import TransformerEncoder


def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if ``fn`` can be called with ``name`` as a keyword
            argument.

    Returns:
        bool: whether ``fn`` accepts a ``name`` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )



class Attention_block(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128):
        super(Attention_block,self).__init__()
        self.W_wav = ConvNorm(ain_chan, ain_chan, 1, 1, groups=ain_chan)
        
        self.W_video = ConvNorm(vin_chan, ain_chan, 1, 1)

        self.psi = nn.Sequential(
            ConvNorm(ain_chan, ain_chan, 1, 1, groups=ain_chan),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        
    def forward(self, audio, video):
        audio1 = self.W_wav(audio)
        video1 = self.W_video(torch.nn.functional.interpolate(video, size=audio.shape[-1], mode='nearest'))
        psi = self.relu(audio1+video1)
        psi = self.psi(psi)
        return audio*psi


class ChannelAttention(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128):
        super(ChannelAttention, self).__init__()
        # self.W_wav = ConvNorm(ain_chan, ain_chan, 1, 1, groups=ain_chan)
        self.W_wav = ConvNorm(ain_chan, ain_chan, 1, 1)
        self.W_video = ConvNorm(vin_chan, ain_chan, 1, 1)
        # self.psi = nn.Sequential(
        #     ConvNorm(ain_chan, ain_chan, 1, 1, groups=ain_chan),
        #     nn.Sigmoid()
        # )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.psi = nn.Sequential(
            # ConvNorm(ain_chan, ain_chan, 1, 1, groups=ain_chan),
            nn.Conv1d(ain_chan, ain_chan, 1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, a, v):
        a = self.W_wav(a)
        v = self.W_video(v)
        v = F.interpolate(v, size=a.shape[-1], mode='nearest')
        psi = self.relu(a + v)
        psi = self.gap(psi)        # N,C,T -> N,C,1
        psi = self.psi(psi)
        return a * psi


class ConcatFC2(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128):
        super(ConcatFC2, self).__init__()
        self.W_wav = ConvNorm(ain_chan+vin_chan, ain_chan, 1, 1)
        self.W_video = ConvNorm(ain_chan+vin_chan, vin_chan, 1, 1)

    def forward(self, a, v):
        sa = F.interpolate(a, size=v.shape[-1], mode='nearest')
        sv = F.interpolate(v, size=a.shape[-1], mode='nearest')
        xa = torch.cat([a, sv], dim=1)
        xv = torch.cat([sa, v], dim=1)
        a = self.W_wav(xa)
        v = self.W_video(xv)
        return a, v



class CrossModalAttention(nn.Module):
    def __init__(self, Ca, Cv, Ta, Tv, d=16, num_heads=4):
        super().__init__()
        self.d = d
        self.proj_a = nn.Conv1d(Ca, d, kernel_size=1)
        self.proj_v = nn.Conv1d(Cv, d, kernel_size=1)
        self.reproj_a = nn.Conv1d(d, Ca, kernel_size=1)
        self.reproj_v = nn.Conv1d(d, Cv, kernel_size=1)
        self.a2v = TransformerEncoder(embed_dim=d,
                        num_heads=num_heads,
                        layers=1,
                        attn_dropout=0,
                        relu_dropout=0.1,
                        res_dropout=0.1,
                        embed_dropout=0.25,
                        attn_mask=False)
        self.v2a = TransformerEncoder(embed_dim=d,
                        num_heads=num_heads,
                        layers=1,
                        attn_dropout=0,
                        relu_dropout=0.1,
                        res_dropout=0.1,
                        embed_dropout=0.25,
                        attn_mask=False)

    def forward(self, a, v):
        # a: [B, Ca, Ta],  v: [B, Cv, Tv]
        res_a = a
        res_v = v
        a = self.proj_a(a)
        v = self.proj_v(v)
        a = a.permute(2, 0, 1)      # [B,C,T] -> [T,B,C]
        v = v.permute(2, 0, 1)
        ha = self.a2v(a, v, v)      # [Ta,B,d]
        hv = self.v2a(v, a, a)      # [Tv,B,d]
        ha = ha.permute(1, 2, 0)    # [T,B,C] -> [B,C,T]
        hv = hv.permute(1, 2, 0)
        ha = self.reproj_a(ha)
        hv = self.reproj_v(hv)
        return res_a + ha, res_v + hv


def _repeat(val, times):
    if not isinstance(val, Iterable):
        return [val for _ in times]
    return val


class AudioVisual(nn.Module):
    """
    a(B, N, T) -> [pre_a] ----> [av_part] -> [post] -> (B, n_src, out_chan, T)
                                            /
    v(B, N, T) -> [pre_a]----------/

    [bottleneck]:   -> [layer_norm] -> [conv1d] ->
    [av_part]: Recurrent
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_repeats=8,
        bn_chan=128,
        hid_chan=512,
        upsampling_depth=5,
        norm_type="gLN",
        mask_act="relu",
        act_type="prelu",
        # video
        vin_chan=256,
        vout_chan=256,
        vconv_kernel_size=3,
        # fusion
        fout_chan=256,
        # video frcnn
        video_frcnn=dict(),
        pretrain=None,
        fusion_shared=False,
        audio_interval=1,
        video_interval=1
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.upsampling_depth = upsampling_depth
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.act_type = act_type
        # video part
        self.vin_chan = vin_chan
        self.vout_chan = vout_chan
        self.vconv_kernel_size = vconv_kernel_size
        # fusion part
        self.fout_chan = fout_chan

        self.fusion_shared = fusion_shared

        self.audio_repeats = n_repeats
        self.video_repeats = video_frcnn["iter"]
        self.fusion_repeats = self.video_repeats // video_interval
        self.audio_interval = audio_interval
        self.video_interval = video_interval

        # pre and post processing layers
        self.pre_a = nn.Sequential(normalizations.get(norm_type)(in_chan), nn.Conv1d(in_chan, bn_chan, 1, 1))
        self.pre_v = nn.Conv1d(self.vout_chan, video_frcnn["in_chan"], kernel_size=3, padding=1)
        self.post = nn.Sequential(nn.PReLU(), 
                        nn.Conv1d(fout_chan, n_src * in_chan, 1, 1),
                        activations.get(mask_act)())
        # main modules
        self.video_frcnn = VideoFRCNN(**video_frcnn)
        self.audio_frcnn = FRCNNBlock(bn_chan, hid_chan, upsampling_depth, norm_type, act_type)
        self.audio_concat = nn.Sequential(nn.Conv1d(bn_chan, hid_chan, 1, 1, groups=bn_chan), nn.PReLU())
        self.crossmodal_fusion = self._build_crossmodal_fusion(bn_chan, video_frcnn["in_chan"], self.video_repeats)
        self.init_from(pretrain)


    def _build_crossmodal_fusion(self, ain_chan, vin_chan, repeats=1):
        if self.fusion_shared:
            return ConcatFC2(ain_chan, vin_chan)
        else:
            return nn.ModuleList([
                ConcatFC2(ain_chan, vin_chan) \
                    for _ in range(repeats)])


    def get_crossmodal_fusion(self, i):
        if self.fusion_shared:
            return self.crossmodal_fusion
        else:
            return self.crossmodal_fusion[i]


    def init_from(self, pretrain):
        if pretrain is None:
            return 
        print("Init pre_v and video_frcnn from", pretrain)
        state_dict = torch.load(pretrain, map_location="cpu")["model_state_dict"]

        frcnn_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("module.head.frcnn"):
                frcnn_state_dict[k[18:]] = v
        self.video_frcnn.load_state_dict(frcnn_state_dict)

        pre_v_state_dict = dict(
            weight = state_dict["module.head.proj.weight"],
            bias = state_dict["module.head.proj.bias"])
        self.pre_v.load_state_dict(pre_v_state_dict)


    def fuse(self, a, v):
        """
            /----------------\-----------------\ -------------\ 
        a --> [frcnn*] x ia --> [frcnn*] x ia --> ... ----[]---> [frcnn*] -> ... -> 
                                                       /
        v --> [frcnn]  x iv --> [frcnn]  x iv---> ... --/
            \---------------/------------------/ 
        """
        res_a = a
        res_v = v

        ia, iv = 0, 0
        for j in range(self.fusion_repeats):
            # audio blocks
            for _ in range(self.audio_interval):
                if ia > 0:
                    a = self.audio_concat(res_a + a)
                a = self.audio_frcnn(a)
                # print("audio", ia)
                ia += 1
            # video blocks
            for _ in range(self.video_interval):
                if iv > 0:
                    v = self.video_frcnn.get_concat_block(iv)(res_v + v)
                v = self.video_frcnn.get_frcnn_block(iv)(v)
                # print("video", iv)
                iv += 1
            # fusion block
            # print("fusion", j)
            a, v = self.get_crossmodal_fusion(j)(a, v)

        # audio decoder
        while ia < self.audio_repeats:
            a = self.audio_frcnn(self.audio_concat(res_a + a))
            # print("audio", ia)
            ia += 1
        return a


    def forward(self, a, v):
        # a: [4, 512, 3280], v: [4, 512, 50]
        B, _, T = a.size()
        a = self.pre_a(a)
        v = self.pre_v(v)
        a = self.fuse(a, v)
        a = self.post(a)
        return a.view(B, self.n_src, self.out_chan, T)


    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "n_src": self.n_src,
            "out_chan": self.out_chan,
            "an_repeats": self.an_repeats,
            "fn_repeats": self.fn_repeats,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "upsampling_depth": self.upsampling_depth,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "act_type": self.act_type,
            "vin_chan": self.vin_chan,
            "vout_chan": self.vout_chan,
            "vconv_kernel_size": self.vconv_kernel_size,
            "fout_chan": self.fout_chan,
        }
        return config


class AVFRCNN2Async(BaseAVEncoderMaskerDecoder):
    """
    wav         --> [encoder] ----> [*] --> [decoder] -> [reshape] ->
         \                         /
          [masker]----------------/
         /
    mouth_emb
    """

    def __init__(
        self,
        n_src=1,
        out_chan=None,
        n_repeats=8,
        bn_chan=128,
        hid_chan=512,
        norm_type="gLN",
        act_type="prelu",
        mask_act="sigmoid",
        upsampling_depth=5,
        # video
        vin_chan=256,
        vout_chan=256,
        vconv_kernel_size=3,
        # fusion intervals
        audio_interval=1,
        video_interval=1,
        # fusion
        fout_chan=256,
        # enc_dec
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        video_frcnn=dict(),
        pretrain=None,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        encoder = _Padder(encoder, upsampling_depth=upsampling_depth, kernel_size=kernel_size)
        
        # Update in_chan
        masker = AudioVisual(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            norm_type=norm_type,
            mask_act=mask_act,
            act_type=act_type,
            upsampling_depth=upsampling_depth,
            # video
            vin_chan=vin_chan,
            vout_chan=vout_chan,
            vconv_kernel_size=vconv_kernel_size,
            # fusion
            fout_chan=fout_chan,
            video_frcnn=video_frcnn,
            pretrain=pretrain,
            audio_interval=audio_interval,
            video_interval=video_interval
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)

    # def train(self, mode=True):
    #     super().train(mode)
    #     if mode:    # freeze BN stats
    #         for m in self.modules():
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()



class _Padder(nn.Module):
    def __init__(self, encoder, upsampling_depth=4, kernel_size=21):
        super().__init__()
        self.encoder = encoder
        self.upsampling_depth = upsampling_depth
        self.kernel_size = kernel_size

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(
            self.kernel_size // 2, 2 ** self.upsampling_depth
        )

        # For serialize
        self.filterbank = self.encoder.filterbank
        self.sample_rate = getattr(self.encoder.filterbank, "sample_rate", None)

    def forward(self, x):
        x = pad(x, self.lcm)
        return self.encoder(x)


def pad(x, lcm: int):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padding = torch.zeros(
            list(appropriate_shape[:-1]) + [lcm - values_to_pad],
            dtype=x.dtype,
            device=x.device
        )
        padded_x = torch.cat([x, padding], dim=-1)
        return padded_x
    return x