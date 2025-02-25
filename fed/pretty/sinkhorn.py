import torch
import copy
import numpy as np
import torch.functional as F
from ..pretty import log

class Sinkhorn(object):
    cost_matrix: torch.Tensor
    log_Q: torch.Tensor  # log assignment matrix
    u: torch.Tensor  # row scaling variables
    v: torch.Tensor  # column scaling variables
    dist: torch.Tensor  # log class upper bounds
    rho: float  # allocation fraction
    reg: float  # regularization coefficient
    update_tol: float
    def __init__(
            self,
            client: int,
            num_samples: int,
            indices: list,
            dist: torch.Tensor,
            reg: float,
            max_iters:int,
            tol: float,
            device='cpu'):
        self.client = client
        self.num_samples = num_samples
        self.indices = indices
        self.num_classes = len(dist)
        self.cost_matrix = torch.zeros(self.num_samples,
                                        self.num_classes, device=device)
        self.dist = dist
        self.reg = reg
        self.tol = tol
        self.max_iters = max_iters
        self.device = device
        self.reset()

    def reset(self, ):
        """ reset
        """
        self.cost_matrix = torch.zeros(self.num_samples,
                                        self.num_classes, device=self.device)
        self.maps = {self.indices[i]:i for i in range(len(self.indices))}

    def set_dist(self, dist):
        self.dist = dist

    def get(self, preds, idxs):
        """ get updated probs of current batch of unlabeled data
        """
        indices = [self.maps[i] for i in idxs]
        self.update_cost(preds, idxs)
        P, status = self.solve()
        return P[indices], status

    def update_cost(self, preds, idxs):
        indices = [self.maps[i] for i in idxs]
        for i, p in enumerate(preds):
            self.cost_matrix[indices[i]] = p

    def solve(self,):
        """ solving the OT problem
        """
        P = copy.deepcopy(self.cost_matrix).to(self.device)
        N = P.shape[0]
        c = (torch.ones((N, 1)) / N).to(self.device) # [Nx1]
        r = self.dist.to(self.device) # [Kx1]
        P = torch.pow(P, self.reg)  # [NxK]
        err, _iter = np.inf, 0
        b = copy.deepcopy(c)
        for i in range(self.max_iters):
            if err < self.tol:
                break
            a = r / (P.T @ b)  # (KxN)@(Nx1) = Kx1
            b = c / (P @ a)  # (NxK)@(Kx1) = Nx1
            loss = torch.sum(a) + torch.sum(b)
            if torch.isnan(loss):
                log.verbose(f'nan solution from client: {self.client}')
                return self.cost_matrix, True
            T = N * (a.T * (P * b))
            err = (torch.abs(T.sum(0) - r).sum() / r.sum()).cpu()
            _iter += 1
        P = N * (a.T * (P * b))
        log.verbose(f'client: {self.client}, sinkhorn iter:{_iter}')
        return P.detach(), False
