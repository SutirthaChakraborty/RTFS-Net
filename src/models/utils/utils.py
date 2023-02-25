import torch
import torch.nn as nn

from torch.autograd import Variable


def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, block_stride)).type(input.type()).to(input.device)
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = input[:, :, :-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    block2 = input[:, :, block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    block = torch.cat([block1, block2], 3).view(batch_size, dim, -1, block_size).transpose(2, 3)

    return block.contiguous(), rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size * 2)  # B, N, K, L

    input1 = input[:, :, :, :block_size].contiguous().view(batch_size, dim, -1)[:, :, block_stride:]
    input2 = input[:, :, :, block_size:].contiguous().view(batch_size, dim, -1)[:, :, :-block_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T
