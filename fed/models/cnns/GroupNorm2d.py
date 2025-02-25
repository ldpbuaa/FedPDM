import torch
import torch.nn as nn

class GroupNorm2d(nn.Module):
    def __init__(self, in_features, in_groups, epsilon=1e-5):
        super(GroupNorm2d, self).__init__()
        self.in_groups = in_groups
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1,in_features,1,1))
        self.beta = nn.Parameter(torch.zeros(1,in_features,1,1))
    def forward(self, x):
        samples,channels,dim1,dim2 = x.shape
        x = x.view(samples,self.in_groups,-1)
        mean_is = torch.mean(x,dim = -1).unsqueeze(2)
        variance_is = torch.var(x,dim = -1).unsqueeze(2)
        x = (x - mean_is ) / (variance_is + self.epsilon).sqrt()
        x = x.view(samples,channels,dim1,dim2)
        return x * self.gamma + self.beta