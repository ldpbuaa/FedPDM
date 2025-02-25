import sys
import os
import signal
import collections
import time
import torch
import numpy as np
import copy
from torch.optim import Optimizer
import torch.nn.functional as F

from ..pretty import log, unit
from ..utils import (topk, MovingAverage, AccuracyCounter,
                     ema_model_update, build_loss_fn, get_confuse_matrix)
from ..datasets import INFO, datasets_map
from ..models import factory


class DivergeError(ValueError):
    """ Training loss diverged to NaN.  """


class FedAvgProcess(torch.multiprocessing.Process):
    initialized = False
    len_history = 100

    def __init__(
            self, action, mode, in_queue, out_queue,
            create_func, init_func, loss_func, grad_func, round_epilog,
            model_name, dataset_params, scaling, max_rounds,
            lr, lr_decay_rounds=None, lr_decay_factor=1.0,
            lr_scheduler=None,
            weight_decay=0.0, momentum=0.0,
            parallel=False, device=None, log_level='info',
            client_eval=True, grad_clipping_norm=0, logit_adjust_tau=0,
            ):
        super().__init__()
        self.action = action
        self.mode = mode
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.init_func = init_func
        self.loss_func = loss_func
        self.grad_func = grad_func
        self.round_epilog = round_epilog
        self.model_name = model_name
        self.dataset_params = dataset_params
        self.data_transform = dataset_params['data_transform']
        self.scaling = scaling
        self.lr = lr
        self.max_rounds = max_rounds
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_decay_factor = lr_decay_factor
        self.lr_scheduler_type = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.parallel = parallel
        self.device = device
        self.log_level = log_level
        self.client_eval = client_eval
        self.grad_clipping_norm = grad_clipping_norm
        self.logit_adjust_tau = logit_adjust_tau
        self.unlabeled_mu = dataset_params['unlabeled_mu']
        self.rounds = 0
        create_func(self)

    @property
    def id(self):
        return int(self.name.replace(f"{self.action}Process-", ""))

    def start(self):
        if self.parallel:
            super().start()

    def terminate(self):
        if self.parallel:
            super().terminate()

    def run(self):
        if not self.parallel:
            raise RuntimeError(
                'This method should only be called by the child process.')
        while True:
            self.call(self.in_queue.get())

    def call(self, info):
        tag, action, kwargs = info
        result = {
            'status': 'ok',
            'tag': tag,
            'client': kwargs.get('client'),
            'process': self.id,
        }
        try:
            if not self.initialized:
                self.init()
            r = getattr(self, action)(**kwargs)
            result.update(r)
        except Exception as e:  # pylint: disable=broad-except
            result.update({
                'status': 'error',
                'exception': e,
            })
        self.out_queue.put(result)
        sys.stdout.flush()

    def init(self):
        if self.initialized:
            raise RuntimeError('Repeated initialization.')
        self.initialized = True
        if self.parallel:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        # model
        dataset_name = self.dataset_params['name']
        info = INFO[dataset_name]
        self.input_channel = info['shape'][0]
        self.model_params = info['model_params']
        self.model = factory[self.model_name](
            **self.model_params, input_channel=self.input_channel,
            scaling=self.scaling)
        self.num_classes = self.model_params['num_classes']
        # dataloaders
        self.task = info['task']
        self.ntokens = info.get('ntokens')
        Dataset = datasets_map[dataset_name]
        self.dataloaders, self.labeledloaders= Dataset(**self.dataset_params)
        self.print_stats([self.dataloaders, self.labeledloaders])
        self.batch_size = self.dataset_params['batch_size']
        self.test_dataloader = Dataset(
            dataset_name, False, self.batch_size, None, None, None, True,
            data_dir=self.dataset_params['data_dir'])
        # optimizer
        self._init_opt(self.lr)
        self._init_scheduler(rounds=0)
        # lr decay steps

        # others
        self.images_seen = 0
        log.level = self.log_level
        # custom
        self.init_func(self)
        self.model = self.model.to(self.device)
        self.labeled_cm = torch.zeros((self.num_classes,self.num_classes)).to(self.device)

    def print_stats(self, dataloaders):
        for dataloader in dataloaders:
            total = 0
            for i, dl in enumerate(dataloader):
                log.info(f'client:{i}, data stats:{dl.stats}, sum:{torch.sum(torch.tensor(dl.stats))}')
                total += torch.sum(torch.tensor(dl.stats))
            log.info(f'total num of data: {total}')

    def _init_opt(self, lr, reinit=False, fc_scale=0):
        lr_name = 'initial_lr' if reinit else 'lr'
        params = [{'params': self.model.parameters(), lr_name: lr}]
        # if fc_scale:
        #     log.debug(f'customized learning rate {(fc_scale*lr):.5f} for fc layer')
        #     params = [{'params': self.model.features.parameters(), lr_name: lr},
        #               {'params': self.model.fc.parameters(), lr_name: lr*fc_scale}]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    def _init_scheduler(self, rounds, last_epoch=-1):
        self.lr_scheduler = None
        if self.lr_decay_rounds:
            if self.lr_scheduler_type == 'step':
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.lr_decay_rounds,
                    gamma=self.lr_decay_factor, last_epoch=last_epoch)
            if self.lr_scheduler_type == 'cos':
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_decay_rounds, last_epoch=last_epoch)

    def _reinit_opt(self, supervised=True, opt_state_dict=None, rounds=-1):
        if self.mode == 'pure':
            lr = self.sup_lr if supervised else self.unsup_lr
        elif self.mode == 'mix':
            lr = self.lr
        fc_scale = 1
        self._init_opt(lr, reinit=True, fc_scale=fc_scale)
        if supervised and opt_state_dict is not None:
            self.optimizer.load_state_dict(opt_state_dict)
        self._init_scheduler(rounds, last_epoch=rounds)

    def _get_lr(self):
        if self.lr_scheduler:
            return self.lr_scheduler.get_last_lr()[0]
        else:
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def _step(self, data, target, init_state, avg_losses, avg_accs, rounds):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)

        if self.task == 'language':
            output = output.view(-1, self.ntokens)
        loss = self.loss_func(self, self.model, output, target, init_state, rounds)

        if torch.isnan(loss).any():
            raise DivergeError('Training loss diverged to NaN.')
        avg_losses.add(loss)
        train_acc = topk(output, target)[0]
        avg_accs.add(train_acc)
        loss.backward()
        self.grad_func(self, init_state)
        if self.grad_clipping_norm > 0:
            self.grad_clip()
        self.optimizer.step()
        self.images_seen += target.size(0)
        self.labeled_cm += get_confuse_matrix(output, target, self.num_classes, self.device)

    def get_weight(self, client):
        if self.mode == 'pure':
            return {'weight': len(self.dataloaders[client].dataset)}
        else:
            return {'weight': len(self.labeledloaders[client].dataset)}

    def _iterate(self, dataloader, steps):
        step = 0
        while True:
            for data, target, idx in dataloader:
                if step >= steps:
                    return
                yield data, target, idx
                step += 1

    def local_train(self, client, state, rounds, epochs, record_w):
        self.client, self.epochs, self.rounds = client, epochs, rounds
        self._reinit_opt(True, None, rounds)
        begin_time = time.time()
        msg = f'p: {self.id}, ' f'c: {client}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        dataloader = self.dataloaders[client] if self.mode == 'pure' \
                                            else self.labeledloaders[client]
        iters = len(dataloader)
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = rounds - 1
            self.optimizer.zero_grad()
            self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.model.load_state_dict(state)
        flops, steps = 0, 0
        result = {
            'flops.model': flops,
            'flops.total': flops * steps * self.batch_size,
        }
        init_state = {
            k: v.to(self.device, copy=True) for k, v in state.items()}
        self.model.train()
        if self.logit_adjust_tau > 0:
            self.loss_func = self.adjusted_loss(self.client, 'bce')
        try:
            for _ in range(epochs):
                for data, target, _ in dataloader:
                    data = data if self.data_transform == 'default' else data[0]
                    self._step(data, target, init_state, avg_losses, avg_accs, rounds)
                    if record_w is not None:
                        record_w = self.ema_update(record_w)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}
        self.labeled_cm = self.update_conf_mat(self.labeled_cm, epochs, iters)
        result.update({
            'state': {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'opt_state':None,
            'loss': float(avg_losses.mean()),
            'labeled_data_stats':dataloader.stats,
            'pseudo_label_stats':torch.zeros(self.num_classes).int(),
            'labeled_cm':self.labeled_cm.cpu(),
        })
        result = self.append_result(result)
        if self.client_eval:
            top1, top5 = self.validation(self.model, client, rounds)
            result.update({
                'eval_acc':top1,
            })
            msg += f', eval:{(top1*100):.2f}%'
        end_time = time.time()
        log.verbose(
            f'{msg}, train: {float(avg_accs.mean()):.2%}, '
            f'ep: {epochs}, '
            f'lr: {self._get_lr():.4f}, model flops: {unit(flops)}, '
            f'loss: {(avg_losses.mean()):.4f}, '
            f'round time:{end_time - begin_time:.2f}s.')
        return result

    def append_result(self, result):
        return result

    def validation(self, model, client, rounds):
        """ eval for local model
        """
        model.eval()
        ac = AccuracyCounter(
            len(self.test_dataloader.dataset), (1, 5),
            task=self.task, ntokens=self.ntokens)
        with torch.no_grad():
            for images, labels, _ in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                ac.add(output, labels)

        return ac.logout()

    def grad_clip(self, ):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self.grad_clipping_norm)

    def ema_update(self, record_w):
        return record_w

    def adjusted_loss(self, client, loss_type, reduction='mean'):
        base_probs = self._base_probs(client, normalized=True).to(self.device)
        return build_loss_fn(base_probs, loss_type=loss_type,
                            tau=self.logit_adjust_tau, reduction=reduction)

    def _base_probs(self, client, normalized=False):
        if self.mode == 'mix':
            counts = torch.tensor(self.labeledloaders[client].stats)
        else:
            counts = torch.tensor(self.dataloaders[client].stats)
        if normalized:
            counts = counts/counts.sum()
        return counts

    def next_data(self, data_iter, dataloader):
        try:
            (inputs_w, inputs_s), targets, indices = next(data_iter)
        except:
            data_iter = iter(dataloader)
            (inputs_w, inputs_s), targets, indices = next(data_iter)
        return (inputs_w, inputs_s), targets, indices

    def update_conf_mat(self, labeled_cm, epoch, iters):
        return labeled_cm / (epoch*iters)