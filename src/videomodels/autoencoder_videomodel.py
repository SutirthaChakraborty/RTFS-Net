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

        self.encoder = EncoderAE(in_channels=self.in_channels, base_channels=self.base_channels, num_layers=self.num_layers)
        self.out_channels = self.encoder.out_chan

        self.encoder.load_state_dict(torch.load(pretrain))
        self.encoder.eval()

    def forward(self, x: torch.Tensor):
        batch, chan, frames, h, w = x.size()
        assert chan == 1

        x = x.view(batch * frames, 1, h, w)  # B, F, H, W -> B*F, 1, H, W

        z = self.encoder.forward(x)  # B*F, 1, H, W -> B*F, C, H', W'

        z = z.view(batch, frames, -1)  # B*F, C, H', W' -> B, F, C*H'*W'

        return z
