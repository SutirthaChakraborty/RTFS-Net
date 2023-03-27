import torch
import torch.nn as nn

from thop import profile

from .autoencoder import EncoderAE


class AEVideoModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 4,
        num_layers: int = 3,
        pretrain: str = None,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(AEVideoModel, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.pretrain = pretrain
        self.is2d = is2d

        self.encoder = EncoderAE(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_layers=self.num_layers,
        )

        self.out_channels = self.encoder.out_channels

        if self.pretrain:
            self.init_from(self.pretrain)

        self.get_MACs()

    def forward(self, x: torch.Tensor):
        batch, chan, frames, h, w = x.size()

        x = x.transpose(1, 2).contiguous().view(batch * frames, chan, h, w)  # B, 1, F, H, W -> B*F, 1, H, W

        z = self.encoder.forward(x)  # B*F, 1, H, W -> B*F, C, H', W'

        if self.is2d:
            z = z.view(batch, frames, self.out_channels, -1)  # B*F, C, H', W' -> B, F, C, H'*W'
            z = z.permute(0, 3, 1, 2).contiguous()  # B, F, C, H'*W' ->  B, H'*W', F, C
        else:
            z = z.view(batch, frames, -1)  # B*F, C, H', W' -> B, F, C*H'*W'
            z = z.transpose(1, 2).contiguous()  # B, F, C*H'*W' ->  B, C*H'*W', F

        return z

    def init_from(self, pretrain):
        self.encoder.load_state_dict(torch.load(pretrain, map_location="cpu"))
        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

    def get_MACs(self):
        batch_size = 1
        seconds = 2
        h, w = 88, 88
        video_input = torch.rand(batch_size, 1, seconds * 25, h, w)

        self.macs = profile(self, inputs=(video_input,), verbose=False)[0] / 1000000
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print("Number of MACs in total: {:,.0f}M".format(self.macs))
        print("Number of trainable parameters: {:,.0f}M".format(self.trainable_params))
        print("Number of non trainable parameters: {:,.0f}M".format(self.non_trainable_params))
