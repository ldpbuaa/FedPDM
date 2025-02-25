import sys
import os
import signal
import collections
import time
import torch
import numpy as np
import copy
import math
from torch.optim import Optimizer
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score as f1_score

from ..pretty import log, unit
from ..utils import (topk, MovingAverage, AccuracyCounter, label_guessing,
                     sharpen, softmax_mse_loss, ema_model_update)
from ..datasets import INFO, datasets_map
from ..models import factory
from .process import DivergeError
from .process import FedAvgProcess

class RSCFedProcess(FedAvgProcess):
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
        super().__init__(action, mode, in_queue, out_queue,
            create_func, init_func, loss_func, grad_func, round_epilog,
            model_name, dataset_params, scaling, max_rounds,
            lr, lr_decay_rounds, lr_decay_factor,
            lr_scheduler,
            weight_decay, momentum,
            parallel, device, log_level,
            client_eval, grad_clipping_norm, logit_adjust_tau)
        # hardcoded hyparams
        self.ema_decay = 0.999
        self.lambda_u = 1
        self.threshold = 0

    def local_train(self, client, state, epochs, rounds, supervised, sup_pretrain, iter_num):
        self._reinit_opt(supervised, None, rounds)
        if self.mode == 'pure':
            if supervised:
                epochs = self.sup_epochs
                return super().local_train(client, state, rounds, epochs, None)
            else:
                return self._unsup_local_train(client, state, rounds, self.unsup_epochs,
                                                self.threshold, iter_num)
        if self.mode == 'mix':
            return self._unsup_local_train(client, state, rounds, epochs,
                                                self.threshold, iter_num)

    def _unsup_local_train(self, client, state, rounds, epochs, \
                threshold, iter_num):
        """ use epochs rather than steps in unsupervised local training
            because local selected training samples are increasing
        """
        self.client, self.epochs, self.rounds = client, epochs, rounds
        self.iter_num = iter_num
        begin_time = time.time()
        msg = f'p: {self.id}, ' f'c: {client}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        avg_us_losses = MovingAverage(self.len_history)
        f1_stats = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        dataloader = self.dataloaders[client]
        labeledloader = self.labeledloaders[client] if self.mode == 'mix' else None
        dataset_size = len(self.dataloaders[client].dataset)
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = rounds - 1
            self.optimizer.zero_grad()
            self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.model.load_state_dict(state)
        self.ema_model = copy.deepcopy(self.model)
        flops, steps = 0, 0
        ld_stats = torch.zeros(self.num_classes, ).int()
        sel_pl_stats = torch.zeros(self.num_classes, ).int()
        pl_cor_stats = torch.zeros(self.num_classes, ).int()
        result = {
            'weight': dataset_size,
            'flops.model': flops,
            'flops.total': flops * steps * self.batch_size,
        }
        init_state = {
            k: v.to(self.device, copy=True) for k, v in state.items()}
        self.model.train()
        try:
            for epoch in range(epochs):
                # if self.mode == 'pure':
                #     sel_pl, pl_cor = self._unsup_epoch(threshold,
                #                 init_state, avg_losses, avg_accs, rounds, iter_num)
                # if self.mode == 'mix':
                sel_pl, pl_cor, lds, f1 = self._semi_epoch(dataloader, labeledloader,
                                    threshold, init_state, avg_losses,
                                    avg_us_losses, avg_accs, rounds, iter_num)
                sel_pl_stats += sel_pl
                pl_cor_stats += pl_cor
                ld_stats += lds
                f1_stats.add(f1)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}
        if self.mode == 'mix':
            labeled_data_stats = ld_stats
        else:
            labeled_data_stats = torch.zeros(self.num_classes).int()
        result.update({
            'state': {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'pseudo_label_stats':self.dataloaders[client].stats,
            'labeled_data_stats': labeled_data_stats.int(),
            'iter_num': self.iter_num,
            'uld_f1': float(f1_stats.mean()),
        })
        result = self.append_result(result)
        if self.client_eval:
            top1, top5 = self.validation(self.ema_model, client, rounds)
            result.update({
                'eval_acc':top1,
            })
            msg += f', ev:{(top1*100):.1f}%'
        end_time = time.time()
        unlabeled_data_size = sum(self.dataloaders[client].stats)
        log.verbose(
            f'{msg}, tr:{float(avg_accs.mean()):.1%}, '
            f's_ls:{float(avg_losses.mean()):.4f}, '
            f'u_ls:{float(avg_us_losses.mean()):.4f}, '
            f'lr:{self._get_lr():.4f}, '
            f'ep:{epochs}, '
            f'cor_pl:{pl_cor_stats.sum()}, '
            f'sel_pl:{sel_pl_stats.sum()}, '
            f'pl_acc:{pl_cor_stats.sum()/sel_pl_stats.sum():.1%}, '
            f'tot:{unlabeled_data_size*epochs}, '
            f'r:{(sel_pl_stats.sum()/(unlabeled_data_size*epochs)):.1%}, '
            f'f1:{f1_stats.mean():.3f}, '
            f'T:{int(end_time - begin_time)}s.')
        self.round_epilog(self,)
        return result

    def _semi_epoch(self, dataloader, labeledloader, threshold, init_state,
                    avg_losses, avg_us_losses, avg_accs,  rounds, iter_num):
        """ TODO
        """
        assert self.data_transform == 'twice', 'RSCFed uses Twice data augmentation'
        ld_stats = torch.zeros(self.num_classes, ).int()
        sel_pl_stats = torch.zeros(self.num_classes, ).int()
        pl_cor_stats = torch.zeros(self.num_classes, ).int()
        data_iter = iter(dataloader)
        steps = len(dataloader)
        probs_list, uld_list = [], []
        for s in range(steps):
            if self.mode == 'mix':
                labeled_iter = iter(labeledloader)
                (images_x, _), labels_x, _ = self.next_data(labeled_iter, labeledloader)
                labels_x = labels_x.to(self.device)
            (images_w, images_s), labels_u, _ = self.next_data(data_iter, dataloader)
            uld_list.append(labels_u)
            labels_u = labels_u.to(self.device)
            images_w = images_w.to(self.device)
            with torch.no_grad():
                guessed = label_guessing(self.ema_model, [images_w])
                sharpened = sharpen(guessed)
            if self.mode == 'mix':
                inputs = torch.cat((images_x, images_s)).to(self.device)
                logits = self.model(inputs)
                logits_x, logits_s = logits[:self.batch_size], logits[self.batch_size:]
                Lx = F.cross_entropy(logits_x, labels_x)
            else:
                Lx = 0
                images_s = images_s.to(self.device)
                logits_s = self.model(images_s)

            pseu = torch.argmax(sharpened, dim=1)
            label = labels_u.squeeze()
            if len(label.shape) == 0:
                label = label.unsqueeze(dim=0)
            # sel_mask = torch.max(sharpened, dim=1)[0] > threshold
            # cor_mask = label[sel_mask] == pseu[sel_mask]
            probs = F.softmax(logits_s, dim=1)
            probs_list.append(probs.cpu())
            train_acc = topk(probs, labels_u)[0]
            avg_accs.add(train_acc)
            Lx = 0
            if self.mode == 'mix':
                Lx = F.cross_entropy(logits_x, labels_x)
            Lu = self.lambda_u * softmax_mse_loss(probs, sharpened).mean()
            avg_losses.add(Lx)
            avg_us_losses.add(Lu)
            loss = Lx + self.lambda_u * Lu

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping_norm > 0:
                self.grad_clip()
            self.optimizer.step()
            alpha = min(1 - 1 / (self.iter_num + 1), self.ema_decay)
            self.ema_model = self.ema_update(self.model.state_dict(), decay=alpha)
            sel_pl, pl_cor = self._cal_stats(probs, labels_u, rounds)
            sel_pl_stats += sel_pl
            pl_cor_stats += pl_cor
            if self.mode == 'mix':
                ld_stats += torch.bincount(labels_x.detach().cpu(),
                                       minlength=self.num_classes).int()
            self.iter_num += 1
        pred_idx = torch.cat(probs_list, dim=0).max(dim=1)[1]
        uld_list = torch.cat(uld_list, dim=0)
        f1 = f1_score(pred_idx, uld_list)
        return sel_pl_stats, pl_cor_stats, ld_stats, f1


    def ema_update(self, model_state, decay):
        return ema_model_update(self.ema_model, model_state, decay=decay)

    def _cal_stats(self, probs, labels_u, rounds):
        sel_pl = torch.zeros(self.num_classes, ).int()
        pl_cor = torch.zeros(self.num_classes, ).int()
        pseu = torch.argmax(probs, dim=1)
        sel_mask = torch.max(probs, dim=1)[0] > self.threshold
        cor_mask = (labels_u == pseu)
        sel_labels = labels_u[sel_mask].cpu()
        cor_labels = labels_u[sel_mask * cor_mask].cpu()
        if len(sel_labels):
            sel_pl = torch.bincount(sel_labels, minlength=self.num_classes)
        if len(cor_labels):
            pl_cor = torch.bincount(cor_labels, minlength=self.num_classes)
        return sel_pl.int(), pl_cor.int()