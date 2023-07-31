import torch.nn as nn

from .layers import ConvNormAct, ConvActNorm, ConvolutionalRNN, FeedForwardNetwork, InjectionMultiSum, RNNProjection, DualPathRNN, BiLSTM2D
from .attention import GlobalAttention, GlobalAttentionRNN, GlobalAttention2D, GlobalGALR, MultiHeadSelfAttention, MultiHeadSelfAttention2D


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
