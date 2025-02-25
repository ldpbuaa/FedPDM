import sys
import argparse
import ast
import numpy as np
import torch


from .models import factory
from .sessions import session_map
from .datasets import INFO
from .pretty import log, Summary



def parse():
    parser = argparse.ArgumentParser(description='Fed')
    parser.add_argument("--action", type=str, default="FedAvg", choices=['FedAvg', 'FedPDM', 'RSCFed', 'CBAFed', 'FedMatch', 'FedIRM'])
    parser.add_argument("--mode", type=str, default="mix", choices=['pure', 'mix'])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=['fashionmnist', 'svhn', 'cifar10', 'cifar100', 'isic', 'office'])
    parser.add_argument("--model", type=str, default='smallcnn', choices=['smallcnn', 'resnet18', 'resnet18_gn'])
    parser.add_argument("--data-dir", type=str, default="~/data")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--verbose2", action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--run-name", type=str, default=None, help="The name of the run")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--split-mode", type=str, default='dirichlet')
    parser.add_argument("--split-alpha", type=float, default=1.)
    parser.add_argument("--num-labeled", type=int, default=0)
    parser.add_argument("--unlabeled-mu", type=int, default=1,
                        help='batch size expansion param for unlabeled data')
    parser.add_argument("--data-transform", type=str, default='default', choices=['default', 'twice', 'dual'])
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drop-last", type=ast.literal_eval, default='True',
                        help='drop the last batch when loading data')
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--equal-epochs", action='store_true')
    parser.add_argument("--train-fraction", type=float, default=1.)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", type=str, default=None, choices=['', 'step', 'cos'])
    parser.add_argument("--lr-decay-rounds", type=int, default=0)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.)
    parser.add_argument("--optimizer-momentum", type=float, default=0.)
    parser.add_argument("--sup-epochs", type=int, default=0)
    parser.add_argument("--unsup-epochs", type=int, default=0)
    parser.add_argument("--num-sup-clients", type=int, default=0)
    parser.add_argument("--client-eval", type=ast.literal_eval, default='True')
    parser.add_argument("--matching", type=ast.literal_eval, default='False',
                        help='distribuiton matching for FedPDM')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-clipping-norm", type=float, default=0)
    parser.add_argument("--logit-adjust-tau", type=float, default=0,
                        help='logit adjust param tau')
    parser.add_argument("--corrupt", type=ast.literal_eval, default='False',
                        help='eval on corrupt test data')
    return parser.parse_args()

def _excepthook(etype, evalue, etb):
    # pylint: disable=import-outside-toplevel
    from IPython.core import ultratb
    ultratb.FormattedTB()(etype, evalue, etb)
    for exc in [KeyboardInterrupt, FileNotFoundError]:
        if issubclass(etype, exc):
            sys.exit(-1)
    import ipdb
    ipdb.post_mortem(etb)

def main(args=None):
    # torch.multiprocessing.set_sharing_strategy('file_system') # prone to crash
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn')
    a = args or parse()

    if a.verbose:
        log.level = 'verbose'
    elif a.verbose2:
        log.level = 'debug'
    if a.debug:
        # pylint: disable=import-outside-toplevel
        import debugpy
        port = 5678
        debugpy.listen(port)
        log.info(
            'Waiting for debugger client to attach '
            f'to port {port}... [^C Abort]')
        try:
            debugpy.wait_for_client()
            log.info('Debugger client attached.')
        except KeyboardInterrupt:
            log.info('Abort wait.')
            sys.excepthook = _excepthook
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if a.action in session_map:
        train = session_map[a.action](**vars(a))
        return train.train()
    if a.action == 'info':
        info = INFO[a.dataset]
        model = factory[a.model](
            info['model_params']['num_classes'], scaling=a.model_scaling)
        shape = (a.eval_batch_size, ) + info['shape']
        summary = Summary(model, shape, a.summary_depth)
        return print(summary.format())
    return log.fail_exit(
        f'Unkown action {a.action!r}, accepts: '
        f'info, {", ".join(session_map)}.')