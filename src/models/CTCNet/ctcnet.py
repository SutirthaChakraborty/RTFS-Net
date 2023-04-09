import torch

from thop import profile

from .refinement_module import RefinementModule

from ..layers import ConvNormAct
from ..base_av_model import BaseAVModel

from ...models import encoder, decoder, mask_generator


class CTCNet(BaseAVModel):
    def __init__(
        self,
        n_src: int,
        pretrained_vout_chan: int,
        audio_bn_params: dict,
        video_bn_params: dict,
        enc_dec_params: dict,
        audio_params: dict,
        video_params: dict,
        fusion_params: dict,
        mask_generation_params: dict,
        *args,
        **kwargs,
    ):
        super(CTCNet, self).__init__()

        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params

        self.encoder: encoder.BaseEncoder = encoder.get(self.enc_dec_params["encoder_type"])(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params["upsampling_depth"],
        )
        self.enc_out_chan = self.encoder.out_chan

        self.mask_generation_params["mask_generator_type"] = self.mask_generation_params.get("mask_generator_type", "MaskGenerator")
        self.audio_bn_chan = self.audio_bn_params.get("out_chan", self.enc_out_chan)
        self.audio_bn_params["out_chan"] = self.audio_bn_chan
        self.video_bn_chan = self.video_bn_params["out_chan"]
        self.audio_hid_chan = self.audio_params["hid_chan"]
        self.video_hid_chan = self.video_params["hid_chan"]

        self.audio_bottleneck = ConvNormAct(**self.audio_bn_params, in_chan=self.enc_out_chan)
        self.video_bottleneck = ConvNormAct(**self.video_bn_params, in_chan=self.pretrained_vout_chan)

        self.refinement_module = RefinementModule(
            fusion_params=self.fusion_params,
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
        )

        self.mask_generator: mask_generator.BaseMaskGenerator = mask_generator.get(self.mask_generation_params["mask_generator_type"])(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=self.enc_out_chan,
            bottleneck_chan=self.audio_bn_chan,
            win=self.enc_dec_params.get("win", 0),
        )

        self.decoder: decoder.BaseDecoder = decoder.get(self.enc_dec_params["decoder_type"])(
            **self.enc_dec_params,
            in_chan=self.enc_out_chan * self.n_src if self.mask_generation_params.get("kernel_size", 1) > 0 else self.audio_bn_chan,
            n_src=self.n_src,
        )

        self.get_MACs()

    def forward(self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor):
        audio_mixture_embedding, spec = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, N, T, (F) -> B, C, T, (F)
        video = self.video_bottleneck(mouth_embedding)  # B, N2, T2 -> B, C2, T2

        refined_features = self.refinement_module(audio, video)  # B, C, T, (F) -> B, C, T, (F)

        separated_audios = self.mask_generator(refined_features, audio_mixture_embedding, spec)  # B, C, T, (F) -> B, n_src, N, T, (F)

        separated_audio = self.decoder(separated_audios, audio_mixture.shape)  # B, n_src, N, T, (F) -> B, n_src, L

        return separated_audio

    def get_config(self):
        model_args = {}
        model_args["encoder"] = self.encoder.get_config()
        model_args["audio_bottleneck"] = self.audio_bottleneck.get_config()
        model_args["video_bottleneck"] = self.video_bottleneck.get_config()
        model_args["refinement_module"] = self.refinement_module.get_config()
        model_args["mask_generator"] = self.mask_generator.get_config()
        model_args["decoder"] = self.decoder.get_config()

        return model_args

    def get_MACs(self):
        batch_size = 1
        seconds = 2

        audio_input = torch.rand(batch_size, seconds * 16000)
        if self.video_bn_params.get("is2d", False):
            video_input = torch.rand(batch_size, self.pretrained_vout_chan, seconds * 25, 16)
        else:
            video_input = torch.rand(batch_size, self.pretrained_vout_chan, seconds * 25)

        encoded_audio, context = self.encoder(audio_input)

        bn_audio = self.audio_bottleneck(encoded_audio)
        bn_video = self.video_bottleneck(video_input)

        separated_audio_embedding = self.mask_generator(bn_audio, encoded_audio, context)

        MACs = []

        MACs.append(int(profile(self.encoder, inputs=(audio_input,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.audio_bottleneck, inputs=(encoded_audio,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.audio_bottleneck.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.video_bottleneck, inputs=(video_input,), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.video_bottleneck.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.refinement_module, inputs=(bn_audio, bn_video), verbose=False)[0] / 1000000))
        MACs.append(int(sum(p.numel() for p in self.refinement_module.parameters() if p.requires_grad) / 1000))

        MACs.append(int(profile(self.mask_generator, inputs=(bn_audio, encoded_audio, context), verbose=False)[0] / 1000000))
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
