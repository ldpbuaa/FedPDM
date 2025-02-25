import math
import time
import random
import os

import torch
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import copy

from ..pretty import log, unit
from .base import SessionBase
from .process import DivergeError
from ..utils import (AccuracyCounter, normalize, dist_to_raw)

class CBAFed(SessionBase):
    def __init__(
            self, *args,
            equal_epochs=True,
            num_sup_clients=0,
            sup_epochs=0,
            unsup_epochs=0,
            **kwargs):
        self.equal_epochs = equal_epochs
        self.sup_epochs = sup_epochs
        self.unsup_epochs = unsup_epochs
        # hardcoded hp-params
        self.sup_lr = 0.03
        self.unsup_lr = 0.025
        self.res_con_interval= 5
        self.threshold = 0.95

        super().__init__(*args, **kwargs)
        self.hyperparams += [
            'max_rounds', 'epochs', 'equal_epochs']
        self.num_sup_clients = num_sup_clients if self.mode =='pure' else 0
        self.sup_idx = list(range(self.num_sup_clients))
        self.unsup_idx = list(range(self.num_sup_clients, self.num_clients))
        self.sup_attr = [True for _ in self.sup_idx] + [False for _ in self.unsup_idx]
        # local data number of both sup and unsup clients for global aggregation
        self.client_data_num = torch.zeros(self.num_clients, dtype=torch.int32) #init
        self.pseudo_label_stats = torch.zeros(self.num_classes, dtype=torch.int32) # unsup clients local pseudo label class-wise distribution
        self.label_stats = torch.zeros(self.num_classes, dtype=torch.int32) #
        self.labeled_data_stats = self.labeled_data_statistics()
        self._client_schedule = None
        # history
        self.init_his()
        # hard-coded hyperparams for CBAFed
        self.T_base = 0.84
        self.T_lower = 0.03
        self.T_higher = 0.1
        self.T_upper = 0.9
        self.uld_f1 = torch.zeros(self.num_clients, )

    def process_create(self, process):
        process.equal_epochs = self.equal_epochs
        process.grad_clipping_norm = self.grad_clipping_norm
        process.sup_epochs = self.sup_epochs
        process.unsup_epochs = self.unsup_epochs
        process.sup_lr = self.sup_lr
        process.unsup_lr = self.unsup_lr
        process.res_con_interval = self.res_con_interval
        process.threshold = self.threshold


    def _pack_states(self, clients, rounds):
        states = {c: s for c, s in self.states.items()
                                            if c in clients}
        return states

    def _round(self, rounds, clients, next_clients, epochs, sup_attr):
        """ sup_pretrain=True: all samples clients are purey labeled
        """
        # train
        self.pseudo_label_stats = torch.zeros(self.num_classes, dtype=torch.int32) # reinit it each round
        self.class_confidence = self.class_confident(self.label_stats)
        include_second, second_class, second_h = self.second_class(self.label_stats)
        self.label_stats = torch.zeros(self.num_classes, dtype=torch.int32) # reinit in each round
        states = self._pack_states(clients, rounds)
        #if rounds > 0 and rounds % self.res_con_interval== 0 and not sup_pretrain:
        #    record_w = self.record_w
        #else:
        #    record_w = None
        results, errors = self.async_train(states, epochs, sup_attr,
                                self.class_confidence, include_second, second_class,
                                second_h, self.record_w)
        if not results:
            raise DivergeError('All clients trained to divergence.')
        # aggregate
        states, losses = {}, []
        for c, rv in results.items():
            states[c] = {
                k: v.to(self.state_device) for k, v in rv['state'].items()}
            losses.append(rv['loss'])
            self.client_data_num[c] += sum(rv['pseudo_label_stats'])
            self.client_data_num[c] += sum(rv['labeled_data_stats'])
            if self.mode == 'mix' or (self.model=='pure' and not self.sup_attr[c]):
                self.pseudo_label_stats += rv['pseudo_label_stats']
                self.client_data_num[c] += sum(rv['pseudo_label_stats'])
                self.clt_pl_stats_his[c].append(rv['pseudo_label_stats'])
                self.uld_f1[c] = rv['uld_f1']
        avg_loss = np.mean(losses)
        self.sup_weight_scaling(scale=10)
        weights = normalize({c: self.client_data_num[c] for c in states})
        self.server_state, update_states = self.aggregate(
            self.server_state, states, errors, next_clients, weights)
        self.server_state, update_states = self.res_connect(self.server_state,
                                update_states, next_clients, rounds_interval=self.res_con_interval)
        self.states.update(update_states)
        # info
        self.tb.add_scalar('train/nans', len(errors), rounds)
        # loss
        self.tb.add_scalar('train/loss', avg_loss, rounds)
        # comms
        comms = self.communication_cost(clients, next_clients)
        avg_comms = comms * 2 / (len(clients) + len(next_clients))
        self.metrics['comms'] = self.metrics.get('comms', 0) + comms
        self.tb.add_scalar('train/comms/round/average', avg_comms, rounds)
        self.tb.add_scalar('train/comms/total', self.metrics['comms'], rounds)
        # client accs
        self.tb_client_accs(results, rounds)
        # label dist
        self.label_stats = self.pseudo_label_stats + self.labeled_data_stats
        if self.label_stats.sum() > 0:
            self.tb.add_histogram('train/label_stats', dist_to_raw(self.label_stats),
                              global_step=rounds, max_bins=self.num_classes)
        if self.pseudo_label_stats.sum() > 0:
            self.tb.add_histogram('train/pseudo_label_stats', dist_to_raw(self.pseudo_label_stats),
                              global_step=rounds, max_bins=self.num_classes)
        self.label_stats_his.append(self.label_stats)
        self.pl_stats_his.append(self.pseudo_label_stats)
        self.tb.add_scalar('train/uld_f1', self.uld_f1.mean(), rounds)
        # progress
        info = (
            f'round {rounds}, train loss: '
            f'{avg_loss:.3f}±{np.std(losses) / avg_loss:.1%}, '
            f'comms: {unit(self.metrics["comms"])}B(+{unit(comms)}B)')
        return info

    def train(self):
        start = time.time()
        training = False
        max_rounds = self.max_rounds
        self.record_w = copy.deepcopy(self.server_state)
        save_name = os.path.join(os.path.split(self.checkpoint)[0], f'{self.model_name}_{self.rounds}.pth')
        self.eval(save=True, name=save_name)
        try:
            clients = self._dispense_clients(self.rounds)
            while True:
                begin_time = time.time()
                self.client_data_num = torch.zeros(self.num_clients, dtype=torch.int32)
                # eval
                self.eval(save=training)
                training = True
                self.rounds += 1
                if max_rounds is not None and self.rounds > max_rounds:
                    log.info(
                        f'Max num of rounds ({max_rounds}) reached.')
                    break
                # train
                next_clients = self._dispense_clients(self.rounds)
                info = self._round(self.rounds, clients, next_clients,
                                    self.epochs, self.sup_attr)
                clients = next_clients
                if self.rounds % self.res_con_interval== 0:
                    self.record_w = copy.deepcopy(self.server_state) # record global model for res_connect
                log.info(f'{info}, elapsed: {time.time() - begin_time:.2f}s.')
        except KeyboardInterrupt:
            log.info('Abort.')
        end = time.time()
        total_time = end-start
        return self.finalize(save_his=True, total_time=total_time)

    def _average(self, states, errors=None, server_state=None, weights=None):
        avg_state = {}
        keys = list(states[list(states)[0]])
        weights = weights or normalize(
            {c: self.client_weights[c] for c in states})
        for k in keys:
            s = [s[k].to(self.device) * weights[c] for c, s in states.items()]
            avg_state[k] = sum(s).to(self.state_device)
        return avg_state

    def _duplicate(self, avg_state, next_clients):
        update_states = {}
        for c in next_clients:
            update_states[c] = {
                k: v.detach().clone() for k, v in avg_state.items()}
        return update_states

    def aggregate(self, server_state, states, errors, next_clients, weights):
        avg_state = self._average(states, errors, server_state, weights)
        update_states = self._duplicate(avg_state, next_clients)
        return avg_state, update_states

    def communication_cost(self, clients, next_clients):
        per_client = sum(v.nelement() for v in self.server_state.values())
        return len(clients) * per_client + len(next_clients) * per_client

    def tb_client_accs(self, results, rounds):
        for c, rv in results.items():
            self.tb.add_scalar(f'eval/client_{c}_acc', rv['eval_acc'], rounds)

    def async_train(self, states, epochs, sup_attr,
                    class_confidence, include_second, second_class,
                    second_h, record_w):
        kwargs = []
        for c, s in states.items():
            kw = {
                'client': c,
                'state': s,
                'rounds': self.rounds,
                'epochs': epochs,
                'supervised': sup_attr[c],
                'class_confidence': class_confidence,
                'include_second': include_second,
                'second_class': second_class,
                'second_h': second_h,
                'record_w': record_w,
            }
            kwargs.append(kw)

        results, errors = self.async_call(
            'local_train', kwargs, to_ignore=[DivergeError])
        results = {r.pop('client'): r for r in results}
        if errors:
            errors = {e.pop('client') for e in errors}
            nans = ', '.join(str(e) for e in errors)
            log.error(f'Clients {nans} training diverged to NaN.')
        return results, errors


    def second_class(self, label_stats):
        """ get the second candidate classes and their thresholds
        """
        second_class, second_h = [], []
        label_dists = (label_stats / sum(label_stats))*(self.num_classes / 10) # normalization
        for i in range(len(label_dists)):
            if label_dists[i]<self.T_lower:
                second_class.append(i)
            if label_dists[i]>self.T_higher:
                    second_h.append(i)
        include_second = True if min(label_dists) < self.T_lower else False
        return include_second, second_class, second_h

    def class_confident(self, label_stats):
        """ get class-wise confidence threshold
        """
        label_dists = (label_stats / sum(label_stats))*(self.num_classes / 10) # normalization
        class_confident = label_dists + self.T_base - label_dists.std()
        if self.dataset_name == 'skin' or self.dataset_name == 'SVHN':
            class_confident[class_confident >= 0.9] = 0.9
        else:
            class_confident[class_confident >= self.T_upper] = self.T_upper
        return class_confident

    def res_connect(self, avg_state, update_states, next_clients, rounds_interval=5):
        """ residual connect of CBAFed
        """
        if self.rounds == 1:
            self.record_w = copy.deepcopy(avg_state)
        if self.rounds > 0 and self.rounds % rounds_interval == 0:
            log.info(f'res weight connection {rounds_interval} epoch')
            w_l = {0:self.record_w, 1:copy.deepcopy(avg_state)}
            n_l = [1., 1.]
            avg_state = self._average(w_l, weights=n_l)
            self.record_w = copy.deepcopy(avg_state)
            update_states = self._duplicate(avg_state, next_clients)
        return avg_state, update_states
