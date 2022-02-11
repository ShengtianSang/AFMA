import numpy as np
import torch

class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next result to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
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
            #cm[i]={"TP":0,"FP":0,"TN":0,"FN":0}
            cm[i] = {}
            #去掉一幅图像中不存在某类的ground truth和prediction的计算，如果不加入if这个语句的话，那这种情况的预测iou=1(是错误的)
            #if torch.sum(gt[:, i, :, :]) == 0 :
            #    print("\nground truth 是0")

            #if torch.sum(pr[:, i, :, :]) == 0:
            #    print("\nprediction 是0")

            #if torch.sum(gt[:, i, :, :]) == 0 and torch.sum(pr[:, i, :, :]) == 0:
            #    print("\n出现不存在的情况")
            #    continue

            pred=pr[:, i, :, :]
            target=gt[:, i, :, :]
            cm[i]["TP"] = torch.sum(pred * target).item()
            cm[i]["FP"] = torch.sum(pred * (1 - target)).item()
            cm[i]["FN"] = torch.sum((1 - pred) * target).item()
            cm[i]["TN"] = torch.sum((1 - pred) * (1 - target)).item()

    return cm