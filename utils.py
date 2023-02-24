import os
from collections import defaultdict
import torch
from torchvision import transforms, datasets
import numpy as np


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def compute_multiclass_acc(output, target, topk=1):
    batch_size = target.size(0)
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    correct_k = correct[:topk].view(-1).float().sum(0)
    res.append(correct_k.mul_(1./batch_size))
    return res[0].item()


def compute_multiinst_f1(output, target):
    pred = (torch.sigmoid(output) > 0.5) * 1
    tp = torch.sum((pred == 1) * (target == 1)).item()
    tn = torch.sum((pred == 0) * (target == 0)).item()
    fp = torch.sum((pred == 1) * (target == 0)).item()
    fn = torch.sum((pred == 0) * (target == 1)).item()
    rec = tp / float(tp+fn) if tp+fn > 0 else 0.0
    pre = tp / float(tp+fp) if tp+fp > 0 else 0.0
    if rec == 0 and pre == 0:
        return 0.0
    else:
        return (2.0*rec*pre) / float(rec+pre)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0.
        self.n   = 0.
        return

    def update(self, val, num):
        self.sum += val * num
        self.n += num
        return

    @property
    def avg(self):
        return self.sum / self.n
