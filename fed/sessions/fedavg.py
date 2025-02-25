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
from ..utils import normalize

class FedAvg(SessionBase):
    def __init__(
            self, *args,
            equal_epochs=True,
            num_sup_clients=0,
            **kwargs):
        self.equal_epochs = equal_epochs
        self.num_sup_clients = num_sup_clients
        self._client_schedule = None
        self.sup_lr = 0.03
        self.unsup_lr = 0.02

        super().__init__(*args, **kwargs)
        self.hyperparams += [
            'max_rounds', 'epochs', 'equal_epochs']
        self.sup_idx = list(range(self.num_sup_clients))
        self.unsup_idx = list(range(self.num_sup_clients, self.num_clients))
        self.sup_attr = [True for _ in self.sup_idx] + [False for _ in self.unsup_idx]


    def process_create(self, process):
        process.equal_epochs = self.equal_epochs
        process.fedprox_mu = 0
        process.sup_lr = self.sup_lr
        process.unsup_lr = self.unsup_lr

    @staticmethod
    def process_loss_func(process, model, output, target, state, rounds):
        loss = torch.nn.functional.cross_entropy(output, target)
        if process.fedprox_mu <= 0:
            return loss
        ploss = 0
        for n, p in process.model.named_parameters():
            if p.requires_grad:
                ploss += torch.nn.functional.mse_loss(
                    p, state[n], reduction='sum')
        return loss + 0.5 * process.fedprox_mu * ploss

    def _round(self, rounds, clients, next_clients, epochs):
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
        avg_loss = np.mean(losses)
        self.server_state, update_states = self.aggregate(
            self.server_state, states, errors, next_clients)
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
            clients = clients if self.mode == 'mix' else self.sup_filter(clients)
            while True:
                begin_time = time.time()
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
                next_clients = next_clients if self.mode == 'mix' else self.sup_filter(next_clients)
                info = self._round(self.rounds, clients, next_clients, self.epochs)
                clients = next_clients
                log.info(f'{info}, elapsed: {time.time() - begin_time:.2f}s.')
        except KeyboardInterrupt:
            log.info('Abort.')
        end = time.time()
        total_time = end-start
        return self.finalize(total_time=total_time)

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

    def aggregate(self, server_state, states, errors, next_clients):
        avg_state = self._average(states, errors, server_state)
        update_states = self._duplicate(avg_state, next_clients)
        return avg_state, update_states

    def communication_cost(self, clients, next_clients):
        per_client = sum(v.nelement() for v in self.server_state.values())
        return len(clients) * per_client + len(next_clients) * per_client

    def tb_client_accs(self, results, rounds):
        for c, rv in results.items():
            self.tb.add_scalar(f'eval/client_{c}_acc', rv['eval_acc'], rounds)


    def sup_filter(self, clients):
        sup_clients = []
        for c in clients:
            if self.sup_attr[c]:
                sup_clients.append(c)
        return sup_clients
