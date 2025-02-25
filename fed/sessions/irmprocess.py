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

from ..pretty import log, unit, Sinkhorn, PDM
from ..utils import (topk, MovingAverage, AccuracyCounter, build_loss_fn, to_d,
        label_guessing, sharpen, softmax_mse_loss, ema_model_update, sinkhorn,
        interleave, de_interleave, LinearRampUp, get_confuse_matrix, kd_loss, get_consistency_weight)
from ..datasets import INFO, datasets_map
from ..models import factory
from .process import DivergeError
from .process import FedAvgProcess

class FedIRMProcess(FedAvgProcess):
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
        self.alpha = 0.99 # ema decay of distribution estimation
        self.ema_decay = 0.999 # # ema decay of model update
        self.lamda = 3
        self.rho = 1
        self.threshold = 0.95
        self.lambda_u = 1
        self.T = 1
        self.rampup = LinearRampUp(0, self.max_rounds)
        # sinkhorn
        self.reg = 10
        self.max_iter = 50
        self.tol = 0.005
        self.sn_wu_rounds = 1 # sinkhorn warmup rounds
        self.la_wu_rounds = 1 # logit adjust warmup rounds

    def local_train(self, client, state, opt_state, rounds, epochs, supervised, uld_dist, avg_labeled_cm):
        self._reinit_opt(supervised, opt_state, rounds)
        self.labeled_cm = torch.zeros((self.num_classes, self.num_classes)).to(self.device)
        if self.mode == 'pure':
            if supervised:
                epochs = self.sup_epochs
                return super().local_train(client, state, rounds, epochs, None)
            else:
                return self._unsup_local_train(client, state, rounds, self.unsup_epochs, uld_dist, avg_labeled_cm)
        if self.mode == 'mix':
            return self._unsup_local_train(client, state, rounds, epochs, uld_dist, avg_labeled_cm)

    def _unsup_local_train(self, client, state, rounds, epochs, uld_dist, avg_labeled_cm):
        """ use epochs rather than steps in unsupervised local training
            because local selected training samples are increasing
        """
        self.client, self.epochs, self.rounds = client, epochs, rounds
        begin_time = time.time()
        msg = f'p: {self.id}, ' f'c: {client}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        avg_us_losses = MovingAverage(self.len_history)
        sn_iters = MovingAverage(self.len_history)
        f1_stats = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        dataloader = self.dataloaders[client]
        labeledloader = self.labeledloaders[client] if self.mode == 'mix' else None
        dataset_size = len(dataloader.dataset)
        iters = len(dataloader)
        if self.lr_scheduler:
            # self.lr_scheduler.last_epoch = rounds - 1
            self.optimizer.zero_grad()
            self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.model.load_state_dict(state)
        self.ema_model = copy.deepcopy(self.model)
        self.ori_model = copy.deepcopy(self.model)
        flops, steps = 0, 0 # TODO
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
        # estimate local data distribution
        # dist_train = self.estimate_local_dist(self.model, self.dataloaders[client], self.num_classes)
        # uld_dist = self.ema_local_dist(uld_dist, dist_train)
        # uld_dist = torch.tensor(uld_dist).to(self.device)
        # train local model
        self.model.train()
        self.ema_model.eval()
        self.ori_model.eval()
        try:
            for epoch in range(epochs):
                sel_pl, pl_cor, lds, uld_dist, f1, sims, labeled_cm= self._semi_epoch(dataloader, labeledloader,
                        uld_dist, avg_losses, avg_us_losses, avg_accs, sn_iters, init_state,
                        epoch, rounds, avg_labeled_cm)
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
        self.labeled_cm = self.update_conf_mat(labeled_cm, epochs, iters)
        result.update({
            'state': {
                k: v.detach().clone()
                for k, v in self.ema_model.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'sn_iters': sn_iters.mean(),
            'uld_f1': float(f1_stats.mean()),
            'pseudo_label_stats': sel_pl_stats.int(),
            'labeled_data_stats': labeled_data_stats.int(),
            'pl_cor_stats': pl_cor_stats.int(),
            'uld_dist': uld_dist.detach().cpu(),
            'cost_matrix': None,
            'sims': sims,
            'labeled_cm':self.labeled_cm.cpu(),
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
            f'i:{sn_iters.mean():.2f}, '
            f'T:{int(end_time - begin_time)}s.')
        self.round_epilog(self,)
        return result

    def _semi_epoch(self, dataloader, labeledloader, uld_dist, avg_losses,
                    avg_us_losses, avg_accs, sn_iters, init_state, epoch, rounds,
                    avg_labeled_cm):
        """ semi-supervised training of local clients
        """
        avg_labeled_cm = avg_labeled_cm.to(self.device)
        assert self.data_transform != 'default', 'wrong data transform for semi-supervised training!'
        unlabeled_cm = torch.zeros((self.num_classes,self.num_classes)).to(self.device)
        labeled_cm = torch.zeros((self.num_classes,self.num_classes)).to(self.device)
        ld_stats = torch.zeros(self.num_classes, ).int()
        sel_pl_stats = torch.zeros(self.num_classes, ).int()
        pl_cor_stats = torch.zeros(self.num_classes, ).int()
        probs_list, uld_list = [], []

        self.unsup_loss_func = self.adjusted_loss(self.client, 'ce', reduction='none')
        self.sup_loss_func = self.loss_func
        data_iter = iter(dataloader)
        steps = len(dataloader)
        for s in range(steps):
            if self.mode == 'mix': # in mix mode we have labeled images
                labeled_iter = iter(labeledloader)
                (images_x, _), labels_x, _ = self.next_data(labeled_iter, labeledloader)
                labels_x = labels_x.to(self.device)
            (images_w, images_s), labels_u, u_idxs = self.next_data(data_iter, dataloader)
            uld_list.append(labels_u)
            images_w, labels_u = images_w.to(self.device), labels_u.to(self.device)
            with torch.no_grad():
                logits_w = self.ema_model(images_w)
            if self.mode == 'mix':
                inputs = torch.cat((images_x, images_s)).to(self.device)
                logits = self.model(inputs)
                logits_x, logits_s = logits[:self.batch_size], logits[self.batch_size:]
                Lx = self.sup_loss_func(self, self.model, logits_x, labels_x, init_state, rounds)
                labeled_cm += get_confuse_matrix(logits_x, labels_x, self.num_classes, self.device)
            else:
                Lx = 0
                images_s = images_s.to(self.device)
                logits_s = self.model(images_s)
            probs = torch.softmax(logits_w.detach()/self.T, dim=-1)
            probs_list.append(probs.cpu())
            max_probs, targets_u = torch.max(probs, dim=-1)
            mask = max_probs.ge(self.threshold).float()
            unlabeled_cm += get_confuse_matrix(logits_w[mask.bool()], labels_u[mask.bool()],
                                                            self.num_classes, self.device)
            Lc = kd_loss(avg_labeled_cm, unlabeled_cm) * get_consistency_weight(rounds, self.max_rounds)
            Lu = (self.unsup_loss_func(self, self.model, logits_s,
                                      targets_u, init_state, rounds) * mask).mean()
            loss = Lx +  self.lambda_u * Lu + Lc
            if torch.isnan(loss).any():
                raise DivergeError('Training loss diverged to NaN.')
            train_acc = topk(probs, labels_u)[0]
            avg_accs.add(train_acc)
            avg_losses.add(Lx)
            avg_us_losses.add(self.lambda_u * Lu)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping_norm > 0:
                self.grad_clip()
            self.optimizer.step()
            decay = self.ema_decay
            self.ema_model =  ema_model_update(self.ema_model,
                                self.model.state_dict(), decay=decay)
            sel_pl, pl_cor = self._cal_stats(probs, labels_u, rounds)
            sel_pl_stats += sel_pl
            pl_cor_stats += pl_cor
            if self.mode == 'mix':
                ld_stats += torch.bincount(labels_x.detach().cpu(),
                                       minlength=self.num_classes).int()
        pred_idx = torch.cat(probs_list, dim=0).max(dim=1)[1]
        uld_list = torch.cat(uld_list, dim=0)
        f1 = f1_score(pred_idx, uld_list)
        sims = self.dist_sim(uld_dist, pred_idx)
        return sel_pl_stats, pl_cor_stats, ld_stats, uld_dist, f1, sims, labeled_cm


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


    def unsup_adjusted_loss(self, sel_stats):
        base_probs = (sel_stats/sel_stats.sum()).to(self.device)
        return build_loss_fn(base_probs, loss_type='bce',
                        tau=1, reduction='mean')

    def estimate_local_dist(self, probs_list, num_classes):
        pred_idx = torch.cat(probs_list, dim=0).max(dim=1)[1]
        pred = F.one_hot(pred_idx, num_classes).detach()
        dist = pred.sum(0)
        dist = dist / float(dist.sum())
        return dist.unsqueeze(1)

    def ema_local_dist(self, dist, dist_train):
        dist = dist.unsqueeze(1) if dist.dim()==1 else dist
        dist = dist_train * self.alpha + dist * (1 - self.alpha)
        return dist


    def adjusted_semi_loss(self, client, rounds, uld_dist, loss_type, reduction='mean', mode='same'):
        if self.mode == 'mix':
            labeled_stats = torch.tensor(self.labeledloaders[client].stats)
        else:
            labeled_stats = torch.zeros(self.num_classes)
        unlabeled_counts = torch.tensor(self.dataloaders[client].stats).sum()
        unlabeled_stats = (unlabeled_counts * uld_dist).squeeze()
        if mode == 'same':
            stats = labeled_stats + unlabeled_stats
        elif mode == 'sup':
            stats = torch.tensor(self.labeledloaders[client].stats)
        elif mode == 'unsup':
            stats = torch.tensor(self.dataloaders[client].stats)
        else:
            raise 'unkonwn loss mode!'
        base_probs = (stats/stats.sum()).to(self.device)
        tau = self.rampup.update(rounds) * self.logit_adjust_tau
        return build_loss_fn(base_probs, loss_type=loss_type,
                        tau=tau, reduction=reduction)

    def dist_sim(self, uld_est_dist, pred_idx):
        ''' distribution similarity
        '''
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        unlabeled_stats = torch.tensor(self.dataloaders[self.client].stats)
        unlabeled_dist = unlabeled_stats / unlabeled_stats.sum() # gt dist
        ema_sim = cos(uld_est_dist.squeeze(), unlabeled_dist)
        pred_stats = torch.bincount(pred_idx, minlength=self.num_classes)
        pred_dist = pred_stats / pred_stats.sum()
        pred_sim = cos(pred_dist, unlabeled_dist)
        return [ema_sim, pred_sim]
