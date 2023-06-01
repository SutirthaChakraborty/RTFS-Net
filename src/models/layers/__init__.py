import torch.nn as nn

from .cnn_layers import ConvNormAct, ConvolutionalRNN, FeedForwardNetwork, InjectionMultiSum, RNNProjection
from .attention import GlobalAttention, GlobalAttentionRNN, GlobalAttention2D, GlobalGALR


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
