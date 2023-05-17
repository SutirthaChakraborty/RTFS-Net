import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(2, 5, 20, 20).to(7)
conv = nn.Conv2d(5, 10, 3).to(7)

y = conv(x)
