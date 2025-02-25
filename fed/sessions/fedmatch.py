import math
import time
import random
import os
import copy

import torch
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

from ..pretty import log, unit
from .base import SessionBase
from .process import DivergeError
from ..utils import normalize, build_loss_fn, dist_to_raw

class FedMatch(SessionBase):
    def __init__(
            self, *args,
            equal_epochs=True,
            num_sup_clients=0,
            sup_epochs=0,
            unsup_epochs=0,
            matching=False,
            **kwargs):
        self.equal_epochs = equal_epochs
        self.sup_epochs = sup_epochs
        self.unsup_epochs = unsup_epochs
        self.matching= matching
        self.sup_lr = 0.03
        self.unsup_lr = 0.01

        super().__init__(*args, **kwargs)
        self.num_sup_clients = num_sup_clients if self.mode =='pure' else 0
        self.hyperparams += [
            'max_rounds', 'epochs', 'equal_epochs']
        self._client_schedule = None
        # estimate local unlabeled data distribution
        self.uld_dists = [torch.ones(self.num_classes)/self.num_classes
                                            for _ in range(self.num_clients)]
        self.sup_idx = list(range(self.num_sup_clients))
        self.unsup_idx = list(range(self.num_sup_clients, self.num_clients))
        self.sup_attr = [True for _ in self.sup_idx] + [False for _ in self.unsup_idx]
        self.client_data_num = torch.zeros(self.num_clients, ).int() #init
        self.pseudo_label_stats = torch.zeros(self.num_classes, ).int() # unsup clients local pseudo label class-wise distribution
        self.pl_cor_stats = torch.zeros(self.num_classes, ).int() # statistics of correct pseudo labels
        self.label_stats = torch.zeros(self.num_classes, ).int() #
        self.labeled_data_stats = self.labeled_data_statistics()
        self._client_schedule = None
        self.opt_states = [None for _ in range(self.num_clients)]
        self.cost_matrices = [None for _ in range(self.num_clients)]
        self.sn_iters = torch.zeros(self.num_clients, )
        self.uld_f1 = torch.zeros(self.num_clients, )
        self.ema_sim = torch.zeros(self.num_clients, )
        self.pred_sim = torch.zeros(self.num_clients, )
        self.avg_probs_his = {c:[] for c in range(self.num_clients)}
        # history
        self.init_his()


    def process_create(self, process):
        process.equal_epochs = self.equal_epochs
        process.sup_epochs = self.sup_epochs
        process.unsup_epochs = self.unsup_epochs
        process.sup_lr = self.sup_lr
        process.unsup_lr = self.unsup_lr
        process.matching = self.matching
        process.data_transform = self.data_transform

    @staticmethod
    def process_loss_func(process, model, output, target, state, rounds):
        loss = torch.nn.functional.cross_entropy(output, target)
        return loss

    def _pack_states(self, clients, rounds):
        states = {c: s for c, s in self.states.items()
                                            if c in clients}
        return states

    def _round(self, rounds, clients, next_clients, epochs, sup_attr):
        # train
        states = self._pack_states(clients, rounds)
        results, errors = self.async_train(states, epochs, sup_attr)
        if not results:
            raise DivergeError('All clients trained to divergence.')
        # aggregate
        states, losses = {}, []
        for c, rv in results.items():
            states[c] = {
                k: v.to(self.state_device) for k, v in rv['state'].items()}
            self.additional_results(c, rv)
            losses.append(rv['loss'])
            self.client_data_num[c] += sum(rv['pseudo_label_stats'])
            self.client_data_num[c] += sum(rv['labeled_data_stats'])
            if self.mode == 'mix' or (self.mode =='pure' and not self.sup_attr[c]):
                self.uld_dists[c] = rv['uld_dist']
                self.pseudo_label_stats += rv['pseudo_label_stats']
                self.pl_cor_stats += rv['pl_cor_stats']
                self.clt_local_dist_his[c].append(rv['uld_dist'])
                self.clt_pl_stats_his[c].append(rv['pseudo_label_stats'])
                self.clt_pl_cor_stats_his[c].append(rv['pl_cor_stats'])
                self.cost_matrices[c] = rv['cost_matrix']
                self.sn_iters[c] = rv['sn_iters']
                self.uld_f1[c] = rv['uld_f1']
                self.ema_sim[c] = rv['sims'][0]
                self.pred_sim[c] = rv['sims'][1]
                self.avg_probs_his[c].append(rv['avg_probs_list'])

        avg_loss = np.mean(losses)
        self.sup_weight_scaling(scale=10)
        weights = normalize({c: self.client_data_num[c] for c in states})
        self.server_state, update_states = self.aggregate(
            self.server_state, states, errors, next_clients, weights)
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
        if self.pl_cor_stats.sum() > 0:
            self.tb.add_histogram('train/pl_cor_stats', dist_to_raw(self.pl_cor_stats),
                              global_step=rounds, max_bins=self.num_classes)
        self.label_stats_his.append(self.label_stats)
        self.pl_stats_his.append(self.pseudo_label_stats)
        self.pl_cor_stats_his.append(self.pl_cor_stats)
        self.tb.add_scalar('train/uld_f1', self.uld_f1.mean(), rounds)
        # OT stats
        self.tb.add_scalar('train/OT/sn_iters', self.sn_iters.mean(), rounds)
        self.tb.add_scalar('train/OT/ema_sim', self.ema_sim.mean(), rounds)
        self.tb.add_scalar('train/OT/pred_sim', self.pred_sim.mean(), rounds)

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
        try:
            clients = self._dispense_clients(self.rounds)
            while True:
                begin_time = time.time()
                self.client_data_num = torch.zeros(self.num_clients, ).int()
                self.pseudo_label_stats = torch.zeros(self.num_classes, ).int()
                self.label_stats = torch.zeros(self.num_classes, ).int()
                # eval
                self.eval(save=training)
                training = True
                self.rounds += 1
                if max_rounds is not None and self.rounds > max_rounds:
                    log.info(
                        f'Max num of rounds ({max_rounds}) reached.')
                    break
                # train
                log.info(f'training clients:{clients}')
                next_clients = self._dispense_clients(self.rounds)
                info = self._round(self.rounds, clients, next_clients,
                                        self.epochs, self.sup_attr)
                clients = next_clients
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

    def async_train(self, states, epochs, sup_attr):
        kwargs = []
        for c, s in states.items():
            kw = {
                'client': c,
                'state': s,
                'opt_state': self.opt_states[c],
                'rounds': self.rounds,
                'epochs': epochs,
                'supervised': sup_attr[c],
                'uld_dist': self.uld_dists[c],
                'cost_matrix': self.cost_matrices[c],
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

    def save_hisory(self, ):
        self.save_dist_his(self.pl_stats_his, 'pl_stats_his')
        self.save_dist_his(self.label_stats_his, 'label_stats_his')
        self.save_dist_his(self.clt_pl_stats_his, 'clt_pl_stats_his')
        self.save_dist_his(self.clt_pl_cor_stats_his, 'clt_pl_cor_stats_his')
        self.save_dist_his(self.clt_local_dist_his, 'clt_local_dist_his')
        self.save_dist_his(self.avg_probs_his, 'avg_probs_his')
        self.save_dist_his(self.data_stats(), 'train_data_stats')
