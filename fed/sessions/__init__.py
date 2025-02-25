from .fedavg import FedAvg
from .cbafed import CBAFed
from .rscfed import RSCFed
from .fedpdm import FedPDM
from .fedmatch import FedMatch
from .fedirm import FedIRM

session_map = {
    'FedAvg': FedAvg,
    'FedProx': FedAvg,
    'CBAFed': CBAFed,
    'FedPDM': FedPDM,
    'RSCFed': RSCFed,
    'FedMatch': FedMatch,
    'FedIRM': FedIRM,

}
