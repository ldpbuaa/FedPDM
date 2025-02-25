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
from torcheval.metrics.functional import multiclass_f1_score as f1_score

from ..pretty import log, unit
from ..utils import (topk, MovingAverage, AccuracyCounter,safe_divide, ema_model_update)
from ..datasets import INFO, datasets_map
from ..models import factory
from .process import DivergeError
from .process import FedAvgProcess


class CBAFedProcess(FedAvgProcess):
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
        self.lambda_u = 1


    def local_train(self, client, state, rounds, epochs, supervised, class_confidence,
                    include_second, second_class, second_h, record_w):
        """ supervised: the client training is supervised or not
            sup_pretrain: indicator of supervised pretraining stage
        """
        self._reinit_opt(supervised)
        self.epochs = epochs
        if self.mode == 'pure':
            if supervised:
                epochs = self.sup_epochs
                return super().local_train(client, state, rounds, epochs, record_w)
            else:
                return self.unsup_local_train(client, state, rounds, self.unsup_epochs, \
                            class_confidence, include_second, second_class, second_h, record_w)
        if self.mode == 'mix':
            return self.unsup_local_train(client, state, rounds, self.unsup_epochs, \
                            class_confidence, include_second, second_class, second_h, record_w)

    def unsup_local_train(self, client, state, rounds, epochs, \
                class_confidence, include_second, second_class, second_h, record_w):
        """ use epochs rather than steps in unsupervised local training
            because local selected training samples are increasing
        """
        self.client, self.epochs, self.rounds = client, epochs, rounds
        begin_time = time.time()
        msg = f'p: {self.id}, ' f'c: {client}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        avg_us_losses = MovingAverage(self.len_history)
        f1_stats = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        dataloader = self.dataloaders[client]
        dataset_size = len(dataloader)
        labeledloader = self.labeledloaders[client] if self.mode == 'mix' else None
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = rounds - 1
            self.optimizer.zero_grad()
            self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.model.load_state_dict(state)
        self.ori_model = copy.deepcopy(self.model)
        flops, steps = 0, 0
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
                sel_pl, pl_cor, f1 = self._semi_epoch(dataloader, labeledloader,
                        class_confidence, include_second, second_class,
                        init_state, avg_losses, avg_us_losses, avg_accs, rounds)
                sel_pl_stats += sel_pl
                pl_cor_stats += pl_cor
                f1_stats.add(f1)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}
        # if record_w is not None:
            # self.model = ema_model_update(self.model, record_w)
        if self.mode == 'mix':
            labeled_data_stats = torch.tensor(self.labeledloaders[client].stats)
        else:
            labeled_data_stats = torch.zeros(self.num_classes).int()
        result.update({
            'state': {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'pseudo_label_stats': sel_pl_stats.int(),
            'labeled_data_stats': labeled_data_stats.int(),
            'uld_f1': float(f1_stats.mean()),
        })
        result = self.append_result(result)
        if self.client_eval:
            top1, top5 = self.validation(self.model, client, rounds)
            result.update({
                'eval_acc':top1,
            })
            msg += f', ev:{(top1*100):.1f}%'
        end_time = time.time()
        log.verbose(
            f'{msg}, tr:{float(avg_accs.mean()):.1%}, '
            f's_ls:{float(avg_losses.mean()):.4f}, '
            f'u_ls:{float(avg_us_losses.mean()):.4f}, '
            f'lr:{self._get_lr():.4f}, '
            f'T:{(end_time - begin_time):.1}s.')
        self.round_epilog(self,)
        return result

    def _semi_epoch(self, dataloader, labeledloader, class_confident,
                    include_second, second_c,
                    init_state, avg_losses, avg_us_losses, avg_accs, rounds):
        sel_pl_stats = torch.zeros(self.num_classes, ).int()
        pl_cor_stats = torch.zeros(self.num_classes, ).int()
        pred_all, pred_sec_all, pred_sec_true, probs_all = [], [], [], []
        true_label, train_data, train_label = [], [], []
        sel_pl_sum, total, sel_pl_corr, train_right = 0, 0, 0, 0
        for i, ((image, _), label, _) in enumerate(dataloader):
            image = image.to(self.device)
            # pseudo labeling
            with torch.no_grad():
                total += len(image)
                self.ori_model.eval()
                outputs = self.ori_model(image)
                guessed = F.softmax(outputs, dim=1).cpu()
                probs_all.append(guessed.cpu())
                pseudo_label = torch.argmax(guessed, dim=1).cpu()
                confident_threshold = torch.zeros(pseudo_label.shape)
                for i in range(len(pseudo_label)):
                    confident_threshold[i] = class_confident[pseudo_label[i]]
                sel_pl_corr += torch.sum(label[torch.max(guessed, dim=1)[0] > confident_threshold] == pseudo_label[
                        torch.max(guessed, dim=1)[0] > confident_threshold].cpu()).item()
                train_right += sum([pseudo_label[i].cpu() == label[i].int() for i in range(label.shape[0])])

                pl = pseudo_label[torch.max(guessed, dim=1)[0] > confident_threshold]
                pred_all.append(pseudo_label)
                sel_pl_sum += len(pl)
                select_samples = image[torch.max(guessed, dim=1)[0] > confident_threshold]
                uns_p = guessed[torch.max(guessed, dim=1)[0] <= confident_threshold]
                uns_samples = image[torch.max(guessed, dim=1)[0] <= confident_threshold]
                uns_p_true = label[torch.max(guessed, dim=1)[0] <= confident_threshold]
                if include_second:
                    pl_u = []
                    sample_u = []
                    for i in range(len(uns_p)):
                        p = uns_p[i]
                        p[p.argmax()] = 0
                        if p.argmax() in second_c:
                            pred_sec_all.append(p.argmax())
                            pred_sec_true.append(uns_p_true[i].detach().cpu().numpy())
                            sample_u.append(uns_samples[i].detach().cpu().numpy())
                            pl_u.append(p.argmax())
                    pl_u = torch.tensor(pl_u).long()
                    sample_u = torch.tensor(np.array(sample_u)).to(self.device)
                    if(len(sample_u.shape)==3):
                        sample_u = sample_u.reshape(1, *sample_u.shape)
                train_label.append(pl)
                train_data.append(select_samples)
                if include_second and len(pl_u) != 0:
                    train_data.append(sample_u)
                    train_label.append(pl_u)
                true_label.append(label)
        # calculate stats
        # log.info(f'client: {self.client}, pseudo: correct [{sel_pl_corr}] / ' + \
        #         f'selected [{sel_pl_sum}] = acc: {(safe_divide(sel_pl_corr*100, sel_pl_sum)):.2f}%')
        #log.info(f'client: {self.client}, prediction: correct [{train_right}] / '+ \
        #         f'total [{total}] = acc: {(safe_divide(train_right*100, total)):.2f}%')
        train_data = torch.cat(train_data, dim=0)
        train_label = torch.cat(train_label, dim=0)
        pred_all = torch.cat(pred_all, dim=0)
        probs_all = torch.cat(probs_all, dim=0)
        true_label = torch.cat(true_label, dim=0)
        f1 = f1_score(pred_all, true_label)
        sel_pl, pl_cor = self._cal_stats(probs_all, true_label, rounds)
        sel_pl_stats += sel_pl
        pl_cor_stats += pl_cor
        # model training
        if self.mode == 'mix': # in mix mode we have labeled images
            for (images_x, _), labels_x, _ in labeledloader:
                images_x, labels_x = images_x.to(self.device), labels_x.to(self.device)
                logits_x = self.model(images_x)
                Lx = F.cross_entropy(logits_x, labels_x)
                if torch.isnan(Lx).any():
                    raise DivergeError('Training loss diverged to NaN.')
                avg_losses.add(Lx)
                self.optimizer.zero_grad()
                Lx.backward()
                self.optimizer.step()

        for i in range(0, len(train_data), self.batch_size):
            self.model.train()
            data_batch = train_data[i:min(len(train_data), i+self.batch_size)].to(self.device)
            if(len(data_batch)==1):
                continue
            label_batch = train_label[i:min(len(train_label), i + self.batch_size)].to(self.device)
            outputs = self.model(data_batch)
            if len(label_batch.shape) == 0:
                label_batch = label_batch.unsqueeze(dim=0)
            if len(outputs.shape) != 2:
                outputs = outputs.unsqueeze(dim=0)

            train_acc = topk(outputs, label_batch)[0]
            avg_accs.add(train_acc)

            Lu = F.cross_entropy(outputs, label_batch)
            avg_us_losses.add(self.lambda_u * Lu)
            loss = self.lambda_u * Lu
            if torch.isnan(loss).any():
                raise DivergeError('Training loss diverged to NaN.')
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping_norm > 0:
                self.grad_clip()
            self.optimizer.step()
        return sel_pl_stats, pl_cor_stats, f1

    def ema_update(self, record_w):
        if record_w is not None and self.epochs % self.res_con_interval == 0: # for CBAFed
            self.model = ema_model_update(self.model, record_w, decay=0.8)
        return copy.deepcopy(self.model).to('cpu').state_dict()


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