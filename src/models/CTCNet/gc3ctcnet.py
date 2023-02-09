from ..base_av_model import BaseAVModel
from ..encoder import ConvolutionalEncoder
from ..decoder import ConvolutionalDecoder
from ..bottleneck import AudioBottleneck, VideoBottleneck
from ..mask_generator import MaskGenerator

from .masker import RefinementModule
from .contextcodec import ContextEncoder, ContextDecoder


class GC3CTCNet(BaseAVModel):
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
        group_size=1,
        context_size=24,
        video_context=False,
        *args,
        **kwargs,
    ):
        super(GC3CTCNet, self).__init__()

        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params
        self.group_size = group_size
        self.context_size = context_size
        self.video_context = video_context

        self.audio_embedding_dim = self.enc_dec_params["out_chan"]
        self.audio_bn_chan = self.audio_bn_params["audio_bn_chan"]
        self.video_bn_chan = self.video_bn_params["video_bn_chan"]
        self.audio_hid_chan = self.audio_params["hid_chan"]
        self.video_hid_chan = self.video_params["hid_chan"]

        self.encoder = ConvolutionalEncoder(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params["upsampling_depth"],
        )

        self.audio_bottleneck = AudioBottleneck(**self.audio_bn_params, in_chan=self.audio_embedding_dim)
        self.video_bottleneck = VideoBottleneck(**self.video_bn_params, in_chan=self.pretrained_vout_chan)

        if self.context_size > 1:
            self.audio_context_enc = ContextEncoder(
                in_chan=self.audio_bn_chan,
                hid_chan=self.audio_hid_chan,
                num_group=self.group_size,
                context_size=self.context_size,
                num_layers=2,
                bidirectional=True,
            )
            if self.video_context:
                self.video_context_enc = ContextEncoder(
                    in_chan=self.video_bn_chan,
                    hid_chan=self.video_hid_chan,
                    num_group=self.group_size,
                    context_size=self.context_size,
                    num_layers=2,
                    bidirectional=True,
                )

        self.refinement_module = RefinementModule(
            **self.fusion_params,
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
        )

        if self.context_size > 1:
            self.context_dec = ContextDecoder(
                in_chan=self.audio_bn_chan,
                hid_chan=self.audio_hid_chan,
                num_group=self.group_size,
                context_size=self.context_size,
                num_layers=2,
                bidirectional=True,
            )

        self.mask_generator = MaskGenerator(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=self.audio_embedding_dim,
            bottleneck_chan=self.audio_bn_chan,
        )

        self.decoder = ConvolutionalDecoder(
            **self.enc_dec_params,
            in_chan=self.enc_dec_params["out_chan"] * self.n_src,
            n_src=self.n_src,
        )

    def __apply_masks(self, masks, audio_mixture_embedding):
        batch_size, _, meta_frames = audio_mixture_embedding.shape
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)
        separated_audio_embedding = separated_audio_embedding.view(batch_size, self.audio_embedding_dim, meta_frames)
        return separated_audio_embedding

    def forward(self, audio_mixture, mouth_embedding):
        audio_mixture_embedding = self.encoder(audio_mixture)

        audio = self.audio_bottleneck(audio_mixture_embedding)
        video = self.video_bottleneck(mouth_embedding)

        # context encoding
        if self.context_size > 1:
            audio, res, squeeze_rest = self.audio_context_enc(audio)
            if self.video_context:
                video = self.video_context_enc(video)[0]

        refined_features = self.refinement_module(audio, video)

        # context decoding
        if self.context_size > 1:
            refined_features = self.context_dec(refined_features, res, squeeze_rest)

        masks = self.mask_generator(refined_features)
        separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

        separated_audio = self.decoder(separated_audio_embedding, audio_mixture.shape)

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
