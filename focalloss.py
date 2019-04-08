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
        y = one_hot(target, input.size(-1), self.device)
        #print('input', input)
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        #print('logit', logit)

        loss = -self.alpha * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum()