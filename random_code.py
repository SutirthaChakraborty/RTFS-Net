import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import BiLSTM2D
from src.models.encoder import STFTEncoder

win = 256
hop_length = 128
chan = 64

audio_len = 32000

window = torch.hann_window(win).to(0)

x = torch.rand((4, audio_len)).to(0)

encoder = STFTEncoder(win, hop_length, chan, act_type="ReLU").to(0)
time_bilstm = BiLSTM2D(chan, chan, 3, 8).to(0)
freq_bilstm = BiLSTM2D(chan, chan, 4, 8).to(0)


x = encoder(x)
x = time_bilstm(x)
x = freq_bilstm(x)

print(torch.prod(torch.tensor(x.shape)))
x = x.permute(0, 3, 1, 2).contiguous().view(4 * x.shape[-1], -1, x.shape[-2])

x = F.unfold(x.unsqueeze(-1), (8, 1), stride=1)
print(torch.prod(torch.tensor(x.shape)))
