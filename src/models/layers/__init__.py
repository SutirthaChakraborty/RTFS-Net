import torch.nn as nn

from .cnn_layers import ConvNormAct, ConvolutionalRNN, FeedForwardNetwork
from .rnn_layers import TAC, RNNProjection
from .attention import GlobalAttention, GlobalAttention2D


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
