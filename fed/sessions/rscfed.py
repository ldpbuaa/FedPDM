import math
import time
import random
import os

import torch
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

from ..pretty import log, unit
from .base import SessionBase
from .process import DivergeError
from ..utils import normalize, model_dist

class RSCFed(SessionBase):
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
        self.unsup_lr = 0.021
        self.meta_client_num = 5
        self.meta_rounds = 3
        self.w_mul_times = 6
        self.sup_scale = 100

        super().__init__(*args, **kwargs)
        self.num_sup_clients = num_sup_clients if self.mode =='pure' else 0
        self.hyperparams += [
            'max_rounds', 'epochs', 'equal_epochs']
        self.sup_idx = list(range(self.num_sup_clients))
        self.unsup_idx = list(range(self.num_sup_clients, self.num_clients))
        self.sup_attr = [True for _ in self.sup_idx] + [False for _ in self.unsup_idx]
        self.iter_num_list = np.zeros(self.num_clients)
        self.meta_states = {}
        self.client_data_num = torch.zeros(self.num_clients, dtype=torch.int32) #init
        self.uld_f1 = torch.zeros(self.num_clients, )
        # history
        self.init_his()



    def process_create(self, process):
        process.equal_epochs = self.equal_epochs
        process.grad_clipping_norm = self.grad_clipping_norm
        process.sup_epochs = self.sup_epochs
        process.unsup_epochs = self.unsup_epochs
        process.sup_lr = self.sup_lr
        process.unsup_lr = self.unsup_lr


    def _round(self, rounds, meta_round, clients, next_clients, epochs):
        # train
        states = self._pack_states(clients, rounds)
        results, errors = self.async_train(states, epochs)
        if not results:
            raise DivergeError('All clients trained to divergence.')
        # aggregate
        states, losses, kd_losses = {}, [], []
        for c, rv in results.items():
            states[c] = {
                k: v.to(self.state_device) for k, v in rv['state'].items()}
            self.additional_results(c, rv)
            losses.append(rv['loss'])
            self.client_data_num[c] += sum(rv['pseudo_label_stats'])
            self.client_data_num[c] += sum(rv['labeled_data_stats'])
            if self.mode == 'mix' or (self.model=='pure' and not self.sup_attr[c]):
                self.iter_num_list[c] = rv['iter_num']
                self.uld_f1[c] = rv['uld_f1']
        avg_loss = np.mean(losses)
        data_stats = {c: self.client_data_num[c] for c in states}
        self.sup_weight_scaling(scale=self.w_mul_times)
        weights = normalize({c: self.client_data_num[c] for c in states})
        avg_state_tmp, _ = self.aggregate(
            self.server_state, states, errors, next_clients, weights)
        reweights = self._distance_reweight(clients, states, avg_state_tmp, weights, data_stats)
        avg_state_meta, _ = self.aggregate(
            self.server_state, states, errors, next_clients, reweights)
        self.states.update(states) # in meta round we do not update client states with global state
        self.meta_states[meta_round] = avg_state_meta
        # comms
        comms = self.communication_cost(clients, next_clients)
        avg_comms = comms * 2 / (len(clients) + len(next_clients))
        self.metrics['comms'] = self.metrics.get('comms', 0) + comms
        if meta_round > 0 and (meta_round+1) % self.meta_rounds == 0:
            # info
            self.tb.add_scalar('train/nans', len(errors), rounds)
            # loss
            self.tb.add_scalar('train/loss', avg_loss, rounds)
            self.tb.add_scalar('train/comms/round/average', avg_comms, rounds)
            self.tb.add_scalar('train/comms/total', self.metrics['comms'], rounds)
            # client accs
            self.tb_client_accs(results, rounds)
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
        try:
            clients = self._dispense_clients()
            while True:
                begin_time = time.time()
                clts_round, self.meta_states = [], {}
                # eval
                self.eval(save=training)
                training = True
                self.rounds += 1
                if max_rounds is not None and self.rounds > max_rounds:
                    log.info(
                        f'Max num of rounds ({max_rounds}) reached.')
                    break
                # train
                for meta_round in range(self.meta_rounds):
                    clts_round.extend(clients)
                    next_clients = self._dispense_clients()
                    info = self._round(self.rounds, meta_round, clients, next_clients,
                                            self.epochs)
                    clients = next_clients
                meta_weights = {c: torch.tensor(1/self.meta_rounds).to(self.device)
                                                for c in range(self.meta_rounds)}
                self.server_state, update_states, = self.aggregate(self.server_state,
                                    self.meta_states, None, clients, meta_weights)
                self.states.update(update_states)
                log.info(f'{info}, elapsed: {time.time() - begin_time:.2f}s.')
        except KeyboardInterrupt:
            log.info('Abort.')
        end = time.time()
        total_time = end-start
        return self.finalize(save_his=True, total_time=total_time)

    def async_train(self, states, epochs):
        kwargs = []
        for c, s in states.items():
            kw = {
                'client': c,
                'state': s,
                'rounds': self.rounds,
                'epochs': epochs,
                'supervised': self.sup_attr[c],
                'sup_pretrain': False,
                'iter_num': self.iter_num_list[c],
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

    def _dispense_clients(self, count=None):
        out = random.sample(list(range(0, self.num_clients)), self.meta_client_num)
        return sorted(out)

    def _distance_reweight(self, clients, states, avg_state_tmp, weights, data_stats):
        dist_list = {c: model_dist(states[c], avg_state_tmp) for c in clients}
        reweights = { c:np.exp(-dist_list[c] * self.sup_scale / data_stats[c]) * weights[c]
                    for c in clients}
        for c in clients:
            if self.sup_attr[c]:
                reweights[c] *= self.w_mul_times
        reweights = normalize(reweights)
        return reweights

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