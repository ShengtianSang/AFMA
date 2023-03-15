import numpy as np
import torch

class Meter(object):

    def reset(self):

        pass

    def add(self, value):

        pass

    def value(self):

        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

def confusion_matrix_fn(pr, gt):
    cm={}
    pr=torch.argmax(pr,dim=-3)

    pr_temp = torch.zeros(gt.shape).to(pr.device)
    src=torch.ones(gt.shape).to(pr.device)

    pr=pr_temp.scatter(dim=1, index=pr.unsqueeze(1), src=src)

    for i in range(pr.size()[1]-1):
        if i not in cm:

            cm[i] = {}

            pred=pr[:, i, :, :]
            target=gt[:, i, :, :]
            cm[i]["TP"] = torch.sum(pred * target).item()
            cm[i]["FP"] = torch.sum(pred * (1 - target)).item()
            cm[i]["FN"] = torch.sum((1 - pred) * target).item()
            cm[i]["TN"] = torch.sum((1 - pred) * (1 - target)).item()

    return cm
