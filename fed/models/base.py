from torch import nn



class ModelBase(nn.Module):
    name = None
    input_size = None

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
