import torch
import numpy as np
import torch.nn.functional as F

class PDM(object):
    def __init__( self,client, num_examples, indices, dist, reg, max_iters,
                                                tol, cost_matrix, device='cpu'):
        self.client = client
        self.num_examples = num_examples
        self.indices = indices
        self.num_classes = len(dist)
        self.cost_matrix = cost_matrix.to(device) if cost_matrix is not None else \
                                         torch.zeros(self.num_examples + 1,
                                        self.num_classes + 1, device=device)
        self.u = torch.zeros(self.num_examples + 1, device=device)
        self.v = torch.zeros(self.num_classes + 1, device=device)
        self.dist = dist.squeeze().to(device)
        self.reg = reg
        self.tol = tol
        self.max_iters = max_iters
        self.device = device
        if cost_matrix is None:
            self.reset()
        self.maps = {self.indices[i]:i for i in range(len(self.indices))}
        self.log_Q = F.log_softmax(-self.reg * self.cost_matrix, -1)

    def set_dist(self, dist):
        self.dist = dist.squeeze().to(self.device)

    def reset(self):
        self.u.zero_()
        self.v.zero_()
        self.cost_matrix[:-1, :-1] = np.log(self.num_classes)

    def get(self, preds, idxs):
        """ get refined probs of current batch of unlabeled data
        """
        log_p = F.log_softmax(preds, -1)
        self.update_cost(log_p, idxs)
        err, iters = self.solve()
        z = self.v.repeat(log_p.shape[0], 1)
        z[:, :-1] += self.reg * log_p
        return F.softmax(z, 1)[:, :-1], err, iters

    def query(self, preds):
        log_p = F.log_softmax(preds, -1)
        z = self.v.repeat(log_p.shape[0], 1)
        z[:, :-1] += self.reg * log_p
        return F.softmax(z, 1)[:, :-1]

    def update_cost(self, log_p, idxs):
        indices = [self.maps[i] for i in idxs]
        self.cost_matrix[indices, :-1] = -log_p.detach()
        log_Q = -self.reg * self.cost_matrix[indices] + self.v.view(1, -1)
        self.u[indices] = -torch.logsumexp(log_Q, 1)
        self.log_Q[indices] = log_Q + self.u[indices].view(-1, 1)

    def solve(self):
        iters, err = 0, np.inf
        mat = -self.reg * self.cost_matrix
        mu = 1 - self.dist.sum()
        rn = 1 + self.num_classes + self.num_examples * (- mu.clamp(max=0))
        c = torch.cat([
            1 + self.num_examples * self.dist,
            1 + self.num_examples * (mu.clamp(min=0).view(-1))])
        while err >= self.tol:
            # col
            log_Q = mat + self.u.view(-1, 1)
            self.v = torch.log(c) - torch.logsumexp(log_Q, 0)
            self.v -= self.v[:-1].mean()
            # row
            log_Q = mat + self.v.view(1, -1)
            self.u = -torch.logsumexp(log_Q, 1)
            self.u[-1] += torch.log(rn)
            self.log_Q = log_Q + self.u.view(-1, 1)
            # err
            err = (torch.abs(self.log_Q.exp().sum(0) - c).sum() / c.sum()).cpu().item()
            iters += 1
            # max iters limit
            if iters >= self.max_iters:
                break
        return err, iters