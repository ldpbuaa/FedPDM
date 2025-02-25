import torch
import numpy as np

from ..utils import use_cuda, default_device
from .table import Table


class Summary:
    def __init__(self, model, input_size=None, summary_depth=None):
        super().__init__()
        self.model = model.to(default_device)
        self.input_size = input_size or ((-1, ) + model.input_size)
        self.summary_depth = summary_depth
        self.summary = {}
        self.info = {}
        self.hooks = []
        self.ignored = []
        self._run()

    def _register_hook(self, module, depth):
        check = not isinstance(module, torch.nn.Sequential)
        check = check and not isinstance(module, torch.nn.ModuleList)
        check = check and (depth is None or depth >= 0)
        check = check and module != self.model
        check = check and getattr(module, 'enable_summary', True)
        if check:
            if depth == 0:
                module.flat_summary = True
            module.register_forward_hook(self._hook)
            depth = depth if depth is None else depth - 1
        if getattr(module, 'flat_summary', False):
            return
        for m in module.children():
            self._register_hook(m, depth)

    def _run(self):
        self._register_hook(self.model, self.summary_depth)
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.model(torch.randn(*self.input_size).type(dtype))
        self._remove()

    def _hook(self, module, inputs, outputs):
        info = self.info[module] = {}
        info['input_shape'] = list(inputs[0].size())
        if isinstance(outputs, (list, tuple)):
            outshape = [list(o.size()) for o in outputs]
        else:
            outshape = list(outputs.size())
        info['output_shape'] = outshape
        info['#params'] = 0
        info['#trainables'] = 0
        if getattr(module, 'flat_summary', False):
            params = list(module.parameters())
        else:
            params = []
            for p in dir(module):
                p = getattr(module, p)
                if isinstance(p, torch.nn.Parameter):
                    params.append(p)
        for p in params:
            num = int(np.product(p.size()))
            info['#params'] += num
            if p.requires_grad:
                info['#trainables'] += num

    def _remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def format(self):
        table = Table(['name', 'shape', '#params'])
        table.add_rule()
        activations = 0
        for name, module in self.model.named_modules():
            try:
                info = self.info[module]
            except KeyError:
                continue
            table.add_row([name, info['output_shape'], info['#params']])
            activations += np.prod(info['output_shape'][1:])
        params = 0
        for p in self.model.parameters():
            params += int(np.product(p.size()))
        table.footer_value('#params', params)
        table.footer_value('shape', f'{activations:,}')
        return table.format()
