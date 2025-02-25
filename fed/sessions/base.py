import os
import copy
import uuid
import queue
import pprint
import datetime
import numpy as np
import random
import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from ..pretty import log, History
from ..utils import default_device, AccuracyCounter
from ..datasets import INFO, datasets_map, corruptions, CorruptDataset
from ..models import factory
from .process import DivergeError, FedAvgProcess
from .cbaprocess import CBAFedProcess
from .pdmprocess import FedPDMProcess
from .rscprocess import RSCFedProcess
from .matchprocess import FedMatchProcess
from .irmprocess import FedIRMProcess


class SessionBase:
    len_history = 100

    def __init__(
            self, action, mode, model, dataset, num_clients, num_shards,
            split_mode, split_alpha=0.5, split_beta=2,
            data_transform='default', num_labeled=0, unlabeled_mu=1,
            model_scaling=1, train_fraction=1,
            learning_rate=0.01, max_rounds=0,
            lr_decay_rounds=300, epochs=20, lr_decay_factor=0.1,
            lr_scheduler=None,
            optimizer_weight_decay=1e-5, optimizer_momentum=0,
            batch_size=50, drop_last=True, eval_batch_size=64,
            num_gpus=None, num_processes=1,
            resume_path=None, run_name=None, device=None,
            data_dir=None,
            vis_plot=False, class_acc_plot=False, split_val=False,
            split_toy_num=False, seed=0,
            grad_clipping_norm = 0, client_eval=True, logit_adjust_tau=0,
            **kwargs):
        super().__init__()
        if kwargs:
            log.info(f'Ignored arguments:\n{pprint.pformat(kwargs)}')
        self.hyperparams = ['lr', 'momentum', 'batch_size']
        self.action = action
        self.mode = mode
        self.data_dir = data_dir
        self.dataset_name = dataset
        self.split_toy_num = split_toy_num
        self.split_val = split_val
        self.split_mode = split_mode
        self.split_alpha = split_alpha
        self.split_beta = split_beta
        self.data_transform = data_transform
        self.num_labeled = num_labeled if self.mode == 'mix' else 0
        self.unlabeled_mu = unlabeled_mu
        self.dataset_params = {
            'name': dataset, 'train': True, 'batch_size': batch_size,
            'num_clients': num_clients, 'num_shards': num_shards,
            'split_mode': split_mode, 'alpha': split_alpha,
            'beta': split_beta, 'parallel': False, 'data_dir': data_dir,
            'num_labeled': self.num_labeled, 'data_transform': data_transform,
            'unlabeled_mu': unlabeled_mu, 'drop_last': drop_last, 'seed': seed,
        }
        self.model_name = model
        self.model_scaling = model_scaling
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lr = learning_rate
        self.max_rounds = max_rounds
        self.lr_decay_rounds = lr_decay_rounds
        self.epochs = epochs
        self.lr_decay_factor = lr_decay_factor
        self.lr_scheduler = lr_scheduler
        self.weight_decay = optimizer_weight_decay
        self.momentum = optimizer_momentum
        self.num_clients = num_clients
        self.train_fraction = train_fraction
        self.num_processes = num_processes
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.num_processes = max(
            num_processes, num_processes - num_processes % self.num_gpus)
        self.parallel = self.num_processes > 1
        self.resume_path = resume_path
        self.device = device or default_device
        self.state_device = 'cpu'
        self.vis_plot = vis_plot
        self.class_acc_plot = class_acc_plot
        self.name = os.path.join(
            self.dataset_name, self.model_name, self.action)
        if run_name:
            self.name = os.path.join(self.name, run_name)
        self.seed = seed
        self.grad_clipping_norm = grad_clipping_norm
        self.client_eval = client_eval
        self.logit_adjust_tau = logit_adjust_tau
        self.hparams = self._init_hparams()
        self._init_model(dataset, model, model_scaling)
        self._init_checkpoint()
        self._init_dataset(dataset, eval_batch_size, data_dir)
        self._init_clients()

    def _init_hparams(self,):
        if self.split_mode == 'dirichlet':
            split_params = f'alp={self.split_alpha}'
        elif self.split_mode == 'quantity':
            split_params = f'beta={self.split_beta}'
        else:
            raise 'Unknow Split Mode!'
        return f'exp-{self.action}-{self.dataset_name}-{self.model_name}'+ \
            f'-bs={self.batch_size}-om={self.momentum}-lr={self.lr}'+ \
            f'-wd={self.weight_decay}'+ \
            f'-epr={self.epochs}-nc={self.num_clients}' + \
            f'-tf={self.train_fraction}-{split_params}-seed={self.seed}'

    def _init_model(self, dataset, model, scaling):
        info = INFO[dataset]
        self.input_shape = info['shape']
        self.model_params = info['model_params']
        self.num_classes = self.model_params['num_classes']
        self.input_channel = info['shape'][0]
        self.model = factory[model](**self.model_params, input_channel=self.input_channel,
                                                                    scaling=scaling)
        self.process_init_func(self)
        self.model = self.model.to(self.device)

    def _init_checkpoint(self,):
        # checkpoint
        self.checkpoint = os.path.join('checkpoints', self.name, f'{self.model_name}.pth')
        os.makedirs(os.path.split(self.checkpoint)[0], exist_ok=True)
        if self.resume_path:
            self._init_checkpoint_resume()
        else:
            self._init_checkpoint_fresh()
        # history
        self.tb = History(self.tbname)

    def _init_checkpoint_resume(self):
        info = torch.load(self.resume_path)
        self.server_state = info['server_state']
        self.states = info['states']
        self.best = info['metric']
        self.metrics = info['metrics']
        self.rounds = info['rounds']
        self.tbname = info['history_name']
        log.info(
            f'Resumed from {self.resume_path!r} at {self.rounds} rounds '
            f'with {info["description"]}.')

    def _init_checkpoint_fresh(self):
        self.server_state = {
            k: v.to('cpu', copy=True)
            for k, v in self.model.state_dict().items()}
        self.states = {
            c: copy.deepcopy(self.server_state)
            for c in range(self.num_clients)}
        self.best = None
        self.metrics = {}
        self.rounds = 0
        dtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.tbname = os.path.join(self.name, dtime)

    def _init_dataset(self, dataset, eval_batch_size, data_dir):
        info = INFO[dataset]
        self.task = info['task']
        self.ntokens = info.get('ntokens')
        Dataset = datasets_map[dataset]
        self.dataloaders, self.labeledloaders =  Dataset(**self.dataset_params)
        self.test_dataloader = Dataset(
            dataset, False, eval_batch_size, None, None, None, True,
            data_dir=data_dir)

    def init_subset(self, Dataset):
        """ init a sub dataset from test set
        """
        self.sub_dataset_params = copy.deepcopy(self.dataset_params)
        self.sub_dataset_params['train'] = False
        self.sub_dataset_params['loader_name'] = 'sub_dataset'
        subset_indices = []
        num_images = 100
        for j in range(self.num_classes):
            indices = [i for i, label in
                        enumerate(self.test_dataloader.dataset.targets)
                        if label == j]
            subset_indices += indices[:num_images]
        self.sub_dataset_params.update({'subset': subset_indices})
        return Dataset(**self.sub_dataset_params)

    def _init_clients(self):
        qm = torch.multiprocessing if self.parallel else queue
        self._in_queue = qm.Queue()
        self._out_queue = qm.Queue()
        self._processes = []
        for i in range(self.num_processes):
            device = torch.device(
                f'cuda:{i % self.num_gpus}' if self.num_gpus else 'cpu')
            p = eval(f'{self.action}Process')(
                self.action, self.mode, self._out_queue, self._in_queue, self.process_create,
                self.process_init_func, self.process_loss_func,
                self.process_grad_func, self.process_round_epilog,
                self.model_name, self.dataset_params,
                self.model_scaling, self.max_rounds, self.lr, self.lr_decay_rounds,
                self.lr_decay_factor, self.lr_scheduler, self.weight_decay,
                self.momentum, self.parallel, device, log.level, self.client_eval,
                self.grad_clipping_norm, self.logit_adjust_tau)
            p.daemon = True
            self._processes.append(p)
        for p in self._processes:
            p.start()
        log.verbose(
            f'Initialized {self.num_processes} process(es) '
            f'on {self.num_gpus} GPU(s).')
        self._async_flags = set()
        client_kwargs = [{'client': c} for c in range(self.num_clients)]
        weights, _ = self.async_call('get_weight', client_kwargs)
        weights = {w['client']: w['weight'] for w in weights}
        self.client_weights = [v for k, v in sorted(weights.items())]

    def eval(self, save=True, name=None):
        if self.rounds % 10 == 0:
            result = self._eval_corrupt(self.server_state)
        else:
            result = self._eval(self.server_state)
        if self.task == 'image':
            top1, top5 = result
            is_best = self.best is None or top1 > self.best
            info = {
                'metric': top1,
                'metrics': {**self.metrics, 'top1': top1, 'top5': top5},
                'is_best': is_best,
                'description': f'{top1:.2%} top1, {top5:.2%} top5',
            }
            self.tb.add_scalar('eval/top1', top1, self.rounds)
            self.tb.add_scalar('eval/top5', top5, self.rounds)
        elif self.task == 'language':
            is_best = self.best is None or result < self.best
            info = {
                'metric': result,
                'metrics': {**self.metrics, 'entropy': result},
                'is_best': is_best,
                'description': f'loss {result:.3f}',
            }
            self.tb.add_scalar('eval/entropy', result, self.rounds)
        else:
            raise ValueError
        info.update({'history_name': self.tbname})
        self.metrics = info['metrics']
        self.tb.flush()
        text = f'round {self.rounds}: eval accs = {info["description"]}'
        info.update({
            'rounds': self.rounds,
            'states': self.states,
            'server_state': self.server_state,
        })
        if save:
            if info['is_best']:
                self.best = info['metric']
                self.best_metrics = info['metrics']
                self.save_checkpoint(info, self.checkpoint)
                log.info(f'{text}, saved best')
            if name:
                self.save_checkpoint(info, name)
                log.info(f'{text}, {name} saved')
            log.info(text)
        else:
            log.info(text)
        return info

    def _eval(self, state=None):
        if state:
            self.model.load_state_dict(state)
        self.model.eval()
        ac = AccuracyCounter(
            len(self.test_dataloader.dataset), (1, 5),
            task=self.task, ntokens=self.ntokens)
        with torch.no_grad():
            for images, labels, _ in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                ac.add(output, labels)
        return ac.logout()

    def _eval_corrupt(self, state=None):
        if state:
            self.model.load_state_dict(state)
        self.model.eval()
        accs_top1, accs_top5 = [], []
        for corr in corruptions:
            images_file = f'{corr}.npy'
            labels_file = 'labels.npy'
            normalize_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize( mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.247, 0.243, 0.261])
                                                        ])
            dataset = CorruptDataset(self.data_dir, images_file, labels_file,
                                                transform=normalize_transform)
            test_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            ac = AccuracyCounter(
                len(self.test_dataloader.dataset), (1, 5),
                task=self.task, ntokens=self.ntokens)
            with torch.no_grad():
                for images, labels, _ in test_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    output = self.model(images)
                    ac.add(output, labels)
            top1, top5 = ac.logout()
            accs_top1.append(top1)
            accs_top5.append(top5)
            log.info(f'data corruption: [{corr}] acc: {top1:.2%}, {top5:.2%}')
        return [torch.mean(torch.tensor(accs_top1)),
                torch.mean(torch.tensor(accs_top5))]



    def async_call(
            self, method, kwargs, to_raise=(Exception, ), to_ignore=()):
        # call
        for kw in kwargs:
            tag = uuid.uuid4()
            info = (tag, method, kw)
            if self.parallel:
                self._out_queue.put(info)
            else:
                self._processes[0].call(info)
            self._async_flags.add(tag)
        # wait
        results, errors = [], []
        while self._async_flags:
            r = self._in_queue.get()
            self._async_flags.remove(r.pop('tag'))
            if r['status'] == 'ok':
                results.append(r)
                continue
            errors.append(r)
            e = r['exception']
            if any(isinstance(e, i) for i in to_ignore):
                continue
            if any(isinstance(e, r) for r in to_raise):
                raise e
        if not self._in_queue.empty() or not self._out_queue.empty():
            raise RuntimeError('Unexpected clients remain in queue.')
        return results, errors

    def async_train(self, states, epochs):
        kwargs = []
        for c, s in states.items():
            kw = {
                'client': c,
                'state': s,
                'rounds': self.rounds,
                'epochs': epochs,
                'record_w':None
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

    def process_create(self, process):
        pass

    @staticmethod
    def process_init_func(process):
        pass

    @staticmethod
    def process_loss_func(process, model, output, target, state, rounds):
        # default loss function
        # TODO check loss for task == 'language'
        return torch.nn.functional.cross_entropy(output, target)

    @staticmethod
    def process_grad_func(process, init_state):
        pass

    @staticmethod
    def process_round_epilog(process,):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self, hparams=None, save_his=False, total_time=None):
        hparams = hparams or self.hyperparams
        hp = {k: getattr(self, k) for k in hparams}
        log.verbose(f'Hyperparameters:\n{pprint.pformat(hp)}')
        log.verbose(f'Metrics:\n{pprint.pformat(self.metrics)}')
        final_metrics = {f'final/{k}': v for k, v in self.metrics.items()}
        self.tb.add_hparams(hp, final_metrics)
        if save_his:
            self.save_hisory()
        total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
        log.info(f'Finished with total time:{total_time}')
        return self.metrics

    def save_checkpoint(self, info, name):
        torch.save(info, name)

    def additional_results(self, *args, **kwargs):
        pass

    def labeled_data_statistics(self,):
        """ get label statistics of supervised clients
        """
        labeled_data_stats = torch.zeros(self.num_classes)
        if self.mode == 'pure':
            for c in self.sup_idx:
                labeled_data_stats += torch.tensor(self.dataloaders[c].stats)
        else:
            for c in range(self.num_clients):
                labeled_data_stats += torch.tensor(self.labeledloaders[c].stats)
        return labeled_data_stats.int()

    def sup_weight_scaling(self, scale=10):
        """ scaling the data weight of supervised clients
        """
        for c in range(self.num_clients):
            if self.sup_attr[c]:
                self.client_data_num[c] *= scale

    def save_dist_his(self, dist, name):
        save_path = os.path.join(os.path.split(self.checkpoint)[0], f'{name}.pth')
        torch.save(dist, save_path)

    def init_his(self, ):
        self.clt_pl_stats_his = {c:[] for c in range(self.num_clients)}
        self.clt_pl_cor_stats_his = {c:[] for c in range(self.num_clients)}
        self.clt_local_dist_his = {c:[] for c in range(self.num_clients)}
        self.pl_stats_his = []
        self.label_stats_his = []
        self.pl_cor_stats_his = []

    def _pack_states(self, clients, rounds):
        states = {c: s for c, s in self.states.items()
                                            if c in clients}
        return states

    def save_hisory(self, ):
        raise NotImplementedError


    def _dispense_clients(self, rounds, count=None):
        if count is None:
            count = max(1, int(self.num_clients * self.train_fraction))
        out = []
        for _ in range(count):
            if not self._client_schedule:
                self._client_schedule = list(range(self.num_clients))
                random.shuffle(self._client_schedule)
            out.append(self._client_schedule.pop())
        return sorted(out)

    def data_stats(self, ):
        data_stats = {}
        dataloaders = {'unlabeled':self.dataloaders,
                       'labeled': self.labeledloaders}
        for n, dl in dataloaders.items():
            if dl:
                stats = {c: torch.tensor(dl[c].stats)
                                        for c in range(self.num_clients)}
                data_stats[n] = stats
        return data_stats