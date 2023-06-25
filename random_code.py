import torch
import torch.nn as nn
import torch.nn.functional as F


win = 32
hop_length = 16

# win = 256
# hop_length = 128

audio_len = 32000

window = torch.hann_window(win).to(0)

x = torch.rand((4, audio_len)).to(0)

spec = torch.stft(x, n_fft=win, hop_length=hop_length, window=window, return_complex=True)
spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F

print(spec.shape)
print(spec.shape[-1] * spec.shape[-2])

spec = torch.complex(spec[:, 0], spec[:, 1])  # B*n_src, T, F
spec = spec.transpose(1, 2).contiguous()

output = torch.istft(spec, n_fft=win, hop_length=hop_length, window=window, length=audio_len)  # B*n_src, L
