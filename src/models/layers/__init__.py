from .cnn_layers import ConvNormAct, ConvNormAct2D
from .rnn_layers import GC_RNN, TAC
from .frcnn import FRCNN
from .tdanet import TDANet
from .attention import GlobalAttention


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
