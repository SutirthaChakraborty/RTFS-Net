import torch
import torch.nn as nn
from .autoencoder import EncoderAE


class AEVideoModel(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_layers: int, pretrain: str = None):
        super(AEVideoModel, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.pretrain = pretrain

        self.encoder = EncoderAE(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_layers=self.num_layers,
        )

        self.out_channels = self.encoder.out_chan
        if self.pretrain:
            self.init_from(self.pretrain)

    def forward(self, x: torch.Tensor):
        batch, chan, frames, h, w = x.size()

        x = x.transpose(1, 2).contiguous().view(batch * frames, chan, h, w)  # B, F, H, W -> B*F, 1, H, W

        z = self.encoder.forward(x)  # B*F, 1, H, W -> B*F, C, H', W'

        z = z.view(batch, frames, -1)  # B*F, C, H', W' -> B, F, C*H'*W'

        return z

    def init_from(self, pretrain):
        self.encoder.load_state_dict(torch.load(pretrain, map_location="cpu"))
        self.encoder.eval()
