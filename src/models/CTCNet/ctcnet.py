import torch

from thop import profile

from ...models import encoder
from ...models import decoder

from ..layers import ConvNormAct
from ..base_av_model import BaseAVModel
from ..mask_generator import MaskGenerator

from .refinement_module import RefinementModule
from .contextcodec import ContextEncoder, ContextDecoder


class CTCNet(BaseAVModel):
    def __init__(
        self,
        n_src: int,
        pretrained_vout_chan: int,
        gc3_params: dict,
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
        self.gc3_params = gc3_params

        self.audio_bn_chan = self.audio_bn_params["out_chan"]
        self.video_bn_chan = self.video_bn_params["out_chan"]
        self.audio_hid_chan = self.audio_params["hid_chan"]
        self.video_hid_chan = self.video_params["hid_chan"]

        self.encoder = encoder.get(self.enc_dec_params["encoder_type"])(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params["upsampling_depth"],
        )
        self.enc_out_chan = self.encoder.out_chan

        self.audio_bottleneck = ConvNormAct(**self.audio_bn_params, in_chan=self.enc_out_chan)
        self.video_bottleneck = ConvNormAct(**self.video_bn_params, in_chan=self.pretrained_vout_chan)

        self.audio_context_enc = ContextEncoder(
            in_chan=self.audio_bn_chan,
            hid_chan=self.audio_hid_chan,
            gc3_params=self.gc3_params["audio"],
        )

        self.video_context_enc = ContextEncoder(
            in_chan=self.video_bn_chan,
            hid_chan=self.video_hid_chan,
            gc3_params=self.gc3_params["video"],
        )

        self.refinement_module = RefinementModule(
            fusion_params=self.fusion_params,
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
            gc3_params=self.gc3_params,
        )

        self.context_dec = ContextDecoder(
            in_chan=self.audio_bn_chan,
            hid_chan=self.audio_hid_chan,
            gc3_params=self.gc3_params["audio"],
        )

        self.mask_generator = MaskGenerator(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=self.enc_out_chan,
            bottleneck_chan=self.audio_bn_chan,
        )

        self.decoder = decoder.get(self.enc_dec_params["decoder_type"])(
            **self.enc_dec_params,
            in_chan=self.enc_out_chan * self.n_src,
            n_src=self.n_src,
        )

        self.get_MACs()

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        shape = audio_mixture_embedding.shape
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)
        separated_audio_embedding = separated_audio_embedding.view(shape[0], self.n_src * self.enc_out_chan, *shape[-(len(shape) // 2) :])
        return separated_audio_embedding

    def forward(self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor):
        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, N, T, (F) -> B, C, T, (F)
        video = self.video_bottleneck(mouth_embedding)  # B, N2, T2 -> B, C2, T2

        # context encoding
        audio, res, squeeze_rest = self.audio_context_enc(audio)
        video = self.video_context_enc(video)[0]

        refined_features = self.refinement_module(audio, video)  # B, C, T, (F) -> B, C, T, (F)

        # context decoding
        refined_features = self.context_dec(refined_features, res, squeeze_rest)

        masks = self.mask_generator(refined_features)  # B, C, T, (F) -> B, n_src, N, T, (F)
        separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)  # B, n_src, N, T, (F) -> B, n_src*N, T, (F)

        separated_audio = self.decoder(separated_audio_embedding, audio_mixture.shape)  # B, n_src*N, T, (F) -> B, n_src, L

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
        video_input = torch.rand(batch_size, self.pretrained_vout_chan, seconds * 25)

        encoded_audio = self.encoder(audio_input)

        bn_audio = self.audio_bottleneck(encoded_audio)
        bn_video = self.video_bottleneck(video_input)

        audio, res, squeeze_rest = self.audio_context_enc(bn_audio)

        masks = self.mask_generator(bn_audio)
        separated_audio_embedding = self.__apply_masks(masks, encoded_audio)

        macs = profile(self.encoder, inputs=(audio_input,), verbose=False)[0] / 1000000
        print("Number of MACs in encoder: {:,.0f}M".format(macs))

        macs = profile(self.audio_bottleneck, inputs=(encoded_audio,), verbose=False)[0] / 1000000
        print("Number of MACs in audio BN: {:,.0f}M".format(macs))

        macs = profile(self.video_bottleneck, inputs=(video_input,), verbose=False)[0] / 1000000
        print("Number of MACs in video BN: {:,.0f}M".format(macs))

        macs = profile(self.audio_context_enc, inputs=(bn_audio,), verbose=False)[0] / 1000000
        print("Number of MACs in audio context encoder: {:,.0f}M".format(macs))

        macs = profile(self.video_context_enc, inputs=(bn_video,), verbose=False)[0] / 1000000
        print("Number of MACs in video context encoder: {:,.0f}M".format(macs))

        macs = profile(self.refinement_module, inputs=(audio, bn_video), verbose=False)[0] / 1000000
        print("Number of MACs in RefinementModule: {:,.0f}M".format(macs))

        macs = profile(self.context_dec, inputs=(audio, res, squeeze_rest), verbose=False)[0] / 1000000
        print("Number of MACs in context decoder: {:,.0f}M".format(macs))

        macs = profile(self.mask_generator, inputs=(audio,), verbose=False)[0] / 1000000
        print("Number of MACs in mask generator: {:,.0f}M".format(macs))

        macs = profile(self.decoder, inputs=(separated_audio_embedding, encoded_audio.shape), verbose=False)[0] / 1000000
        print("Number of MACs in decoder: {:,.0f}M".format(macs))

        self.macs = profile(self, inputs=(audio_input, video_input), verbose=False)[0] / 1000000
        print("Number of MACs in total: {:,.0f}M".format(self.macs))
