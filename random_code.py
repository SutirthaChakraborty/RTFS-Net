import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import MLP, Permutator, ConvNormAct
from src.models.utils import get_MACS_params
from src.models.encoder import STFTEncoder

win = 256
hop_length = 128
chan = 64
ks = 3
audio_len = 32000
bs = 1

patch_size = 16
dim = 64
depth = 1

encoder = STFTEncoder(win, hop_length, chan, ks, act_type="ReLU").to(0)

x = torch.rand((bs, audio_len)).to(0)
x = encoder(x)


attention = ConvNormAct(chan, chan, 3, is2d=True).to(0)
att_map = F.softmax(attention(x))

print(att_map.shape)


m, p = get_MACS_params(attention, (x,))
print(f"MACs: {m},", f"parameters: {p}")
