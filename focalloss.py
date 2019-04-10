import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes, device):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).to(device)
        mask = Variable(mask, volatile=index.volatile).to(device)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, device, alpha=1, gamma=0, eps=1e-9):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.device = device

    def forward(self, input, target):
        pt = input * target + (1 - input) * (1 - target)
        pt = pt.clamp(self.eps, 1.0 - self.eps)
        CE = -torch.log(pt)
        FL = CE * (1 - pt) ** self.gamma
        loss = torch.sum(FL, dim=1)

        return loss