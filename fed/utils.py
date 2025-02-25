import math
import collections
import time
import torch
import torch.nn.functional as F
import os
import copy
import numpy as np
from enum import Enum
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torch import nn, Tensor
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from functools import reduce
from typing import Sequence, Tuple

use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
default_device = torch.device('cuda' if use_cuda else 'cpu')


def sparse_kaiming_(tensor, mode='normal', gain=1.0):
    if tensor.dim() < 2:
        raise ValueError(
            'Fan can not be computed for tensor with fewer than 2 dimensions.')
    fan = tensor[0].numel()
    if fan == 0:
        raise ValueError('Cannot initialize an empty tensor.')
    with torch.no_grad():
        if mode == 'normal':
            std = math.sqrt(2.0 * gain / fan)
            return tensor.normal_(0, std), std
        if mode == 'uniform':
            bound = math.sqrt(6.0 * gain / fan)
            return tensor.uniform_(-bound, bound), bound
        raise ValueError(f'Unrecognized mode {mode}.')


def split_by_sizes(values, sizes):
    if sum(sizes) != len(values):
        raise ValueError(
            'The total slice sizes must equal the length of the list.')
    i = 0
    slices = []
    for s in sizes:
        slices.append((i, i + s))
        i += s
    return [values[s:e] for s, e in slices]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def normalize(values):
    if isinstance(values, collections.Mapping):
        normed = {k: v / sum(values.values()) for k, v in values.items()}
        return values.__class__(normed)
    values = list(values)
    return [v / sum(values) for v in values]


def dict_gather(mapping, keys=None):
    """
    >>> mapping = {'a': [0, 1, 2], 'b': [3, 4]}
    >>> dict_gather(mapping)
    ([0, 1, 2, 3, 4], {'a': 3, 'b': 2})
    """
    lens = {}
    vs = []
    for k in keys or mapping:
        v = mapping[k].flatten()
        lens[k] = len(v)
        vs.append(v)
    return torch.cat(vs), lens


def dict_scatter(values, key_lens):
    """
    >>> values = [0, 1, 2, 3, 4]
    >>> key_lens = {'a': 3, 'b': 2}
    >>> dict_scatter(values, key_lens)
    {'a': [0, 1, 2], 'b': [3, 4]}
    """
    values = split_by_sizes(values, key_lens.values())
    return dict(zip(key_lens, values))


def dict_diff(before, after):
    return {k: before[k] - after[k] for k in before}


def dict_filter(
        mapping, *, types=None, keys=None,
        prefix=None, suffix=None, prefixes=(), suffixes=()):
    new_mapping = {}
    for k, v in mapping.items():
        if types is not None and not isinstance(v, types):
            continue
        if keys is not None and k not in keys:
            continue
        if prefix is not None and not k.startswith(prefix):
            continue
        if suffix is not None and not k.endswith(suffix):
            continue
        if prefixes and not any(k.startswith(p) for p in prefixes):
            continue
        if suffixes and not any(k.endswith(s) for s in suffixes):
            continue
        new_mapping[k] = v
    return new_mapping


def topk(output, target, k=(1, ), count=False):
    _, pred = output.topk(max(k), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    batch = 1 if count else target.size(0)
    return [float(correct[:k].sum()) / batch for i, k in enumerate(k)]


def topk_mask(values, k, view):
    threshold = values.topk(k).min()
    return values >= threshold, threshold


def safe_log(value, epsilon=1e-20):
    return torch.log(torch.clamp(value, min=epsilon))


def gumbel(shape, epsilon=1e-20):
    uniform = torch.rand(shape)
    if use_cuda:
        uniform = uniform.to(device)
    return -safe_log(-safe_log(uniform))


def gumbel_softmax(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    output = logits + gumbel(logits.shape)
    return torch.nn.functional.softmax(output / temperature, dim=-1)


def gumbel_topk(probs_or_logits, k, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    if temperature == 0:
        return torch.topk(logits, k)
    if temperature != 1:
        logits /= temperature
    return torch.topk(logits + gumbel(logits.shape, epsilon), k)


def gumbel_max(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    return gumbel_topk(probs_or_logits, 1, temperature, log, epsilon)


def entropy(output, target, ntokens):
    return [torch.nn.CrossEntropyLoss()(output.view(-1, ntokens), target)]


class AccuracyCounter:
    supported_tasks = ['image', 'language']

    def __init__(self, num, k=(1, ), task='image', ntokens=None, num_classes=10):
        super().__init__()
        self.num = num
        self.k = k
        self.correct = [0] * len(k)
        self.entropies = []
        self.size = 0
        if task not in self.supported_tasks:
            raise ValueError(
                f'Task {task!r} not in supprted list {self.supported_tasks}.')
        self.task = task
        self._ntokens = ntokens
        self.num_classes = num_classes
        self.class_accs = {c:0 for c in range(self.num_classes)}

    def add(self, output, target):
        self.size += target.size(0)
        if self.task == 'image':
            for i, a in enumerate(topk(output, target, self.k, True)):
                self.correct[i] += a
            self.per_class_accs(output, target)
        if self.task == 'language':
            self.entropies.append(entropy(output, target, self._ntokens))

    def logout(self):
        if self.task == 'image':
            return self.accuracies()
        if self.task == 'language':
            return self.entropy()
        raise ValueError

    def entropy(self):
        return np.mean(self.entropies)

    def accuracies(self):
        for i in range(len(self.k)):
            yield self.correct[i] / self.size

    def errors(self):
        for a in self.accuracies():
            yield 1 - a

    def progress(self):
        return self.size / self.num

    def per_class_accs(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        for c in range(self.num_classes):
            correct = ((labels==preds)*(labels==c)).sum()
            self.class_accs[c] += correct

class MovingAverage:
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.items = []

    def add(self, value):
        self.items.append(float(value))
        if len(self.items) > self.num:
            self.items = self.items[-self.num:]

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

    def flush(self):
        self.items = []

    def __format__(self, mode):
        text = f'{self.mean():.5f}'
        if 's' not in mode:
            return text
        return text + f'±{self.std() * 100:.2f}%'

    def __float__(self):
        return self.mean()


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self, epoch):
        entries = [f"Evaluation: Epoch[{epoch}]"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def safe_divide(a, b):
    if b == 0:
        return 0
    else:
        return a/b


def build_loss_fn(base_probs, loss_type='ce', tau=1.0, reduction='mean'):
    """Builds the loss function.
    Args:
        base_probs: Base probabilities to use in the logit-adjusted loss.
        tau: Temperature scaling parameter for the base probabilities.
        loss_type: the loss type for training. options:['lc', 'ce', 'bce']
    Returns:
        A loss function with signature loss(labels, logits).
    """
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    def lc_loss_fn(process, model, logits, labels, state, rounds):
        """ logit calibration loss
        """
        base_probs[base_probs==0] = 1 # avoid deviding by zero
        logits = logits - tau * torch.pow(base_probs, -1/4)
        loss = criterion(logits, labels)
        return loss
    def bce_loss_fn(process, model, logits, labels, state, rounds):
        """ balanced cross entropy loss
        """
        logits = logits + tau * torch.log(base_probs + 1e-12) # avoid underflow
        loss = criterion(logits, labels)
        return loss
    def ce_loss_fn(process, model, logits, labels, state, rounds):
        """ cross entropy loss
        """
        loss = criterion(logits, labels)
        return loss
    def pc_loss_fn(process, model, logits, labels, state, rounds):
        """ partial class loss: zero-out logits  from missing classes
            but leave imbalanced classes untouched
        """
        class_filter = (base_probs>=1e-5).int()
        logits = logits * class_filter
        loss = criterion(logits, labels)
        return loss
    loss_maps = {
        'lc': lc_loss_fn,
        'ce': ce_loss_fn,
        'bce': bce_loss_fn,
        'pc': pc_loss_fn,
    }
    return loss_maps[loss_type]

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

class LinearRampUp:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.rampup_len = end - start

    def update(self, cur):
        assert cur >= 0 and self.rampup_len >= 0
        if cur >= self.rampup_len:
            return 1.0
        elif cur < self.start:
            return 0.
        else:
            return (cur - self.start) / self.rampup_len


def sinkhorn(pred, eta, r_in=None, rec=False, device=None):
    PS = pred.detach().to(device)
    K = PS.shape[1]
    N = PS.shape[0]
    PS = PS.T
    c = torch.ones((N, 1)) / N # [Nx1]
    r = r_in.to(device) # [1xK]
    c = c.to(device)
    # average column mean 1/N
    PS = torch.pow(PS, eta)  # K x N
    r_init = copy.deepcopy(r)
    inv_N = 1. / N
    err = 1e6
    # error rate
    _counter = 1
    for i in range(50):
        if err < 1e-1:
            break
        r = r_init * (1 / (PS @ c))  # (KxN)@(N,1) = K x 1
        # 1/K(Plambda * beta)
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        err = torch.sum(c_new) + torch.sum(r)
        # print(f'c_new.shape:{c_new.shape}, err_c:{torch.sum(c_new)}, err_r:{torch.sum(r)}')
        # 1/N(alpha * Plambda)
        if _counter % 10 == 0:
            err = torch.sum(c_new) + torch.sum(r)
            if torch.isnan(err):
                # This may very rarely occur (maybe 1 in 1k epochs)
                # So we do not terminate it, but return a relaxed solution
                print('====> Nan detected, return relaxed solution')
                pred_new = pred + 1e-5 * (pred == 0)
                relaxed_PS, _ = sinkhorn(pred_new, eta, r_in=r_in, rec=True, device=device)
                z = (1.0 * (pred != 0))
                relaxed_PS = relaxed_PS * z
                return relaxed_PS, True
        c = c_new
        _counter += 1
    # 注意PS就是cost matrix，在迭代的过程中不变，以下为了求解结果方便复用了之前的PS
    PS *= torch.squeeze(c)
    PS = PS.T
    PS *= torch.squeeze(r)
    PS *= N
    return PS.detach(), False



def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.detach().to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.detach().to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.detach().to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.detach().to(device)
    return optim

def dist_to_raw(dist):
    """ convert a distribution of data to raw data
    """
    data = []
    for i, d in enumerate(dist):
        data += list(torch.ones(int(d))*i)
    return torch.tensor(data)


def ema_model_update(ema_model, model_state, decay=0.9):
    device = next(ema_model.parameters()).device
    needs_module = hasattr(ema_model, 'module')
    param_keys = [k for k, _ in ema_model.named_parameters()]
    buffer_keys = [k for k, _ in ema_model.named_buffers()] # BN
    with torch.no_grad():
        msd = model_state
        esd = ema_model.state_dict()
        for k in param_keys:
            if needs_module:
                j = 'module.' + k
            else:
                j = k
            model_v = msd[j].to(device)
            ema_v = esd[k]
            esd[k].copy_(ema_v * decay + (1. - decay) * model_v)
            msd[j] = model_v.to('cpu')
        for k in buffer_keys:
            if needs_module:
                j = 'module.' + k
            else:
                j = k
            msd_v = msd[j].to(device)
            # esd[k].copy_(msd[j]) # copy update
            ema_v = esd[k] # ema update
            esd[k].copy_(ema_v * decay + (1. - decay) * msd_v) # ema_update
            msd[j] = msd_v.to('cpu')
    return ema_model


def label_guessing(model: nn.Module, batches: Sequence[Tensor], model_type=None) -> Tensor:
    model.eval()
    with torch.no_grad():
        probs = [F.softmax(model(batch), dim=1) for batch in batches]
        mean_prob = reduce(lambda x, y: x + y, probs) / len(batches)

    return mean_prob

def sharpen(x: Tensor, t=0.5) -> Tensor:
    sharpened_x = x ** (1 / t)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)

def softmax_mse_loss(input_logits, target_logits, softmax=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if softmax:
        input_logits = F.softmax(input_logits, dim=1)
        target_logits = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits-target_logits)**2
    return mse_loss


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cpu() - w_2[key].cpu())
        dist_total += dist.cpu()

    return dist_total.cpu().item()

def to_d(data, device, transform='default'):
    if transform == 'default':
        return data.to(device)
    if transform in ['twice', 'dual']:
        return [d.to(device) for d in data]

def concat(images):
    imgs0 = torch.cat(images[0], dim=0)
    imgs1 = torch.cat(images[1], dim=0)
    return [imgs0, imgs1]


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    batch_size = s[0] // size
    x = x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    l = x[:batch_size]
    u_s = x[batch_size:]
    return l, u_s

# def de_interleave(x, size):
#     s = list(x.shape)
#     batch_size = s[0] // size
#     x = x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
#     l = x[:batch_size]
#     u_w, u_s = x[batch_size:].chunk(2)
#     return l, u_s, u_w

def torch_tile(tensor, dim, n):

    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)

# def get_confuse_matrix(logits, labels, num_class):
#      source_prob = []

#      for i in range(num_class):
#           mask = torch_tile(torch.unsqueeze(labels[:, i], -1), 1, num_class)
#           logits_mask_out = logits * mask
#           logits_avg = torch.sum(logits_mask_out, dim=0) / ( torch.sum(labels[:, i]) + 1e-8 )
#           prob = F.softmax(logits_avg , dim=0)
#           source_prob.append(prob)
#      return torch.stack(source_prob)


from torchmetrics import ConfusionMatrix
def get_confuse_matrix(logits, labels, num_classes, device):
    preds = torch.argmax(logits , dim=1)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    return confmat(preds.cpu(), labels.cpu()).to(device)




def kd_loss(source_matrix, target_matrix):

    Q = source_matrix
    P = target_matrix
    loss = (F.kl_div(Q.log(), P, None, None, 'batchmean') + F.kl_div(P.log(), Q, None, None, 'batchmean'))/2.0
    return loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_consistency_weight(epoch, max_rounds):
     return sigmoid_rampup(epoch, max_rounds)