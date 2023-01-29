from ..base_av_model import BaseAVModel
from ..encoder import ConvolutionalEncoder, PaddedEncoder
from ..decoder import ConvolutionalDecoder
from .masker import Masker


class CTCNet(BaseAVModel):
    def __init__(
        self,
        n_src: int,
        in_chan: int,
        kernel_size: int,
        stride: int,
        hid_chan: int,
        audio_bn_chan: int,
        fusion_repeats: int = 1,
        audio_repeats: int = 3,
        upsampling_depth: int = 5,
        encoder_activation: str = None,
        norm_type: str = "gLN",
        mask_act: str = "sigmoid",
        act_type: str = "prelu",
        video_frcnn: dict = dict(),
        pretrain: str = None,
        audio_shared: bool = True,
        fusion_shared: bool = False,
        fusion_type: str = "ConcatFusion",
        *args,
        **kwargs
    ):
        super().__init__()

        self.n_src = n_src
        self.in_chan = in_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.hid_chan = hid_chan
        self.audio_bn_chan = audio_bn_chan
        self.fusion_repeats = fusion_repeats
        self.audio_repeats = audio_repeats
        self.upsampling_depth = upsampling_depth
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.act_type = act_type
        self.video_frcnn = video_frcnn
        self.pretrain = pretrain
        self.audio_shared = audio_shared
        self.fusion_shared = fusion_shared
        self.fusion_type = fusion_type
        self.encoder_activation = encoder_activation

        encoder = ConvolutionalEncoder(
            in_channels=1,
            out_channels=self.in_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            bias=False,
            encoder_activation=self.encoder_activation,
        )

        self.encoder = PaddedEncoder(
            encoder=encoder,
            in_chan=self.in_chan,
            upsampling_depth=self.upsampling_depth,
            kernel_size=self.kernel_size,
        )

        self.masker = Masker(
            n_src=self.n_src,
            in_chan=self.in_chan,
            hid_chan=self.hid_chan,
            audio_bn_chan=self.audio_bn_chan,
            fusion_repeats=self.fusion_repeats,
            audio_repeats=self.audio_repeats,
            upsampling_depth=self.upsampling_depth,
            norm_type=self.norm_type,
            mask_act=self.mask_act,
            act_type=self.act_type,
            video_frcnn=self.video_frcnn,
            pretrain=self.pretrain,
            audio_shared=self.audio_shared,
            fusion_shared=self.fusion_shared,
            fusion_type=self.fusion_type,
        )

        self.decoder = ConvolutionalDecoder(
            in_channels=self.in_chan * self.n_src,
            out_channels=self.n_src,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            groups=1,
            bias=False,
        )

    def __apply_masks(self, masks, audio_mixture_embedding):
        batch_size, _, meta_frames = audio_mixture_embedding.shape
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)
        separated_audio_embedding = separated_audio_embedding.view(batch_size, self.in_chan, meta_frames)
        return separated_audio_embedding

    def forward(self, audio_mixture, mouth_embedding):
        audio_mixture_embedding = self.encoder(audio_mixture)
        masks = self.masker(audio_mixture_embedding, mouth_embedding)
        separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)
        separated_audio = self.decoder(separated_audio_embedding, audio_mixture.shape)

        return separated_audio

    def get_config(self):

        model_args = {}
        model_args["encoder"] = self.encoder.get_config()
        model_args["decoder"] = self.decoder.get_config()
        model_args["masker"] = self.masker.get_config()

        return model_args
