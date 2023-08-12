import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import MLP, Permutator
from src.models.utils import get_MACS_params
from src.models.encoder import STFTEncoder

win = 256
hop_length = 128
chan = 32
ks = 3
audio_len = 32000
bs = 1

patch_size = 16
dim = 64
depth = 1

encoder = STFTEncoder(win, hop_length, chan, ks, act_type="ReLU").to(0)
if __name__ == "__main__":
    x = torch.rand((bs, audio_len)).to(0)
    x = encoder(x)
    # padding = (
    #     0,
    #     (x.shape[3] // patch_size) * patch_size + patch_size - x.shape[3],
    #     0,
    #     (x.shape[2] // patch_size) * patch_size + patch_size - x.shape[2],
    # )
    # x_pad = nn.functional.pad(x, padding)

    model = MLP(
        image_size=(256, 144),
        in_chan=chan,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
    ).to(0)

    m, p = get_MACS_params(model, (x,))
    print(f"MACs: {m},", f"parameters: {p}")
