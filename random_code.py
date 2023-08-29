import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import MLP, Permutator, ConvNormAct
from src.models.utils import get_MACS_params
from src.models.encoder import STFTEncoder

# win = 256
# hop_length = 128
# chan = 32
# ks = 3
# audio_len = 32000
# bs = 1

# patch_size = 16
# dim = 64
# depth = 1

# encoder = STFTEncoder(win, hop_length, chan, ks, act_type="ReLU").to(0)
# if __name__ == "__main__":
#     x = torch.rand((bs, audio_len)).to(0)
#     x = encoder(x)
#     # padding = (
#     #     0,
#     #     (x.shape[3] // patch_size) * patch_size + patch_size - x.shape[3],
#     #     0,
#     #     (x.shape[2] // patch_size) * patch_size + patch_size - x.shape[2],
#     # )
#     # x_pad = nn.functional.pad(x, padding)

#     model = MLP(
#         image_size=(256, 144),
#         in_chan=chan,
#         patch_size=patch_size,
#         dim=dim,
#         depth=depth,
#     ).to(0)

#     m, p = get_MACS_params(model, (x,))
#     print(f"MACs: {m},", f"parameters: {p}")


from sru import SRU
from sru import SRUpp

length, batch_size, input_size = 5000, 1, 32

x = torch.rand((length, batch_size, input_size * 8)).to(0)

sru_normal = SRU(32 * 8, 32, 1, bidirectional=True).to(0)
m, p = get_MACS_params(sru_normal, (x,))
print(f"MACs: {m},", f"parameters: {p}")
srupp = SRUpp(32 * 8, 32, 32, 1, bidirectional=True).to(0)
m, p = get_MACS_params(srupp, (x,))


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
