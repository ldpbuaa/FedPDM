from .cnns.smallcnn import SmallCNN
from .cnns.resnetcifar import ResNet18_cifar10
from .cnns.resnetcifar_gn import ResNet18_cifar10_GN

factory = {
    'smallcnn': SmallCNN,
    'resnet18': ResNet18_cifar10,
    'resnet18_gn': ResNet18_cifar10_GN
}
