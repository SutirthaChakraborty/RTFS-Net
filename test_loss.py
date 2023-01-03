###
# Author: Kai Li
# Date: 2021-06-22 15:44:09
# LastEditors: Kai Li
# LastEditTime: 2021-06-22 16:38:01
###
import torch

from src.losses import pairwise_neg_sisdr, singlesrc_neg_sisdr, PITLossWrapper

if __name__ == "__main__":
    pit = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    a = torch.randn(2, 1, 32000)
    b = torch.randn(2, 1, 32000)
    print(pit(a, b))
    print(singlesrc_neg_sisdr(a.squeeze(1), b.squeeze(1)))
