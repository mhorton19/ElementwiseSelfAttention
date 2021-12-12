import torch
import torch.nn as nn
import numpy as np

def masked_softmax(matrix, mask, dim=-1):
        exp = torch.exp(matrix)*mask
        return exp/torch.sum(exp, dim, keepdim=True)
