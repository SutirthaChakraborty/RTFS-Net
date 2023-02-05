###
# Author: Kai Li
# Date: 2021-06-21 15:21:58
# LastEditors: Kai Li
# LastEditTime: 2021-09-02 21:25:44
###

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



class Bottomup_Concat_Topdown(nn.Module):
    def __init__(
        self, in_chan=128, out_chan=512, kernel_size=5, upsampling_depth=4, norm_type="gLN", act_type="prelu"
    ):
        super().__init__()
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNorm(
                            out_chan,
                            out_chan,
                            kernel_size=kernel_size,
                            stride=2,
                            groups=out_chan,
                            dilation=1,
                            padding=((kernel_size - 1) // 2) * 1,
                            norm_type=norm_type,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 4, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, residual, bottomup, topdown):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = bottomup[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](bottomup[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(bottomup[i].device),
                    bottomup[i],
                    F.interpolate(bottomup[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(bottomup[i].device),
                    F.interpolate(topdown, size=wav_length, mode="nearest"),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = bottomup[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual


class Bottomup(nn.Module):
    def __init__(
        self, in_chan=128, out_chan=512, kernel_size=5, upsampling_depth=4, norm_type="gLN", act_type="prelu"
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            ConvNorm(
                out_chan,
                out_chan,
                kernel_size=kernel_size,
                stride=1,
                groups=out_chan,
                dilation=1,
                padding=((kernel_size - 1) // 2) * 1,
                norm_type=norm_type,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                ConvNorm(
                    out_chan,
                    out_chan,
                    kernel_size=5,
                    stride=2,
                    groups=out_chan,
                    dilation=1,
                    padding=((kernel_size - 1) // 2) * 1,
                    norm_type=norm_type,
                )
            )

    def forward(self, x):
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        return residual, output[-1], output




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
        an_repeats=4,
        fn_repeats=4,
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
        vn_repeats=5,
        # fusion
        fout_chan=256,
        # video frcnn
        video_frcnn=dict(),
        pretrain=None
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.an_repeats = an_repeats
        self.fn_repeats = fn_repeats
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
        self.vn_repeats = vn_repeats
        # fusion part
        self.fout_chan = fout_chan

        # pre and post processing layers
        self.pre_a = nn.Sequential(normalizations.get(norm_type)(in_chan), nn.Conv1d(in_chan, bn_chan, 1, 1))
        self.pre_v = nn.Conv1d(self.vout_chan, video_frcnn["in_chan"], kernel_size=3, padding=1)
        self.post = nn.Sequential(nn.PReLU(), 
                        nn.Conv1d(fout_chan, n_src * in_chan, 1, 1),
                        activations.get(mask_act)())
        # main modules
        self.video_frcnn = VideoFRCNN(**video_frcnn)
        # self.audio_frcnn = FRCNNBlock(bn_chan, hid_chan, upsampling_depth, norm_type, act_type)
        self.audio_bottomup = Bottomup(bn_chan, hid_chan, 5, upsampling_depth, norm_type, act_type)
        self.audio_topdown = Bottomup_Concat_Topdown(bn_chan, hid_chan, 5, upsampling_depth, norm_type, act_type)
        self.audio_concat = nn.Sequential(nn.Conv1d(bn_chan, hid_chan, 1, 1, groups=bn_chan), nn.PReLU())
        self.crossmodal_fusion = nn.ModuleList([
            ConcatFC2(512, 128) for _ in range(self.an_repeats)
        ])
        self.init_from(pretrain)


    def init_from(self, pretrain):
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
                 o-o
                 o-o\ 
                 o-o-\ 
        v_{t} -> o-o--o -> v_{t+1}
                 \ 
                 o
                 | 
                 o-o
                 o-o\ 
                 o-o-\ 
        a_{t} -> o-o--o -> a_{t+1}
        """
        res_a = a
        res_v = v

        # iter 0
        # a = self.audio_frcnn(a)
        v = self.video_frcnn.frcnn[0](v)
        a, a_out, a_out_list = self.audio_bottomup(a)
        a_out, v = self.crossmodal_fusion[0](a_out, v)
        a = self.audio_topdown(a, a_out_list, a_out)

        # iter 1 ~ self.an_repeats
        assert self.an_repeats <= self.video_frcnn.iter
        for i in range(1, self.an_repeats):
            # video part
            frcnn = self.video_frcnn.frcnn[i]
            concat_block = self.video_frcnn.concat_block[i]
            v = frcnn(concat_block(res_v + v))
            # audio part
            # a = self.audio_frcnn(self.audio_concat(res_a + a))
            a = self.audio_concat(res_a + a)
            a, a_out, a_out_list = self.audio_bottomup(a)
            a_out, v = self.crossmodal_fusion[i](a_out, v)
            a = self.audio_topdown(a, a_out_list, a_out)

        # audio decoder
        for _ in range(self.fn_repeats):
            # a = self.audio_frcnn[-1](self.audio_concat[-1](res_a + a))
            # a = self.audio_frcnn(self.audio_concat(res_a + a))
            a, a_out, a_out_list = self.audio_bottomup(a)
            a = self.audio_topdown(a, a_out_list, a_out)
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
            "vn_repeats": self.vn_repeats,
            "fout_chan": self.fout_chan,
        }
        return config


class AVFRCNN2Thalamus(BaseAVEncoderMaskerDecoder):
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
        an_repeats=4,
        fn_repeats=4,
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
        vn_repeats=5,
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
            an_repeats=an_repeats,
            fn_repeats=fn_repeats,
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
            vn_repeats=vn_repeats,
            # fusion
            fout_chan=fout_chan,
            video_frcnn=video_frcnn,
            pretrain=pretrain
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