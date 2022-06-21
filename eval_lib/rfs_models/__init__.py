from .convnet import convnet4
from .resnet import resnet12
from .resnet import seresnet12
from .wresnet import wrn_28_10
from .dartsmodel import NetworkMiniImageNet
from .augment_cnn import AugmentCNN

from .resnet_new import resnet50

model_pool = [
    'convnet4',
    'resnet12',
    'seresnet12',
    'wrn_28_10',
    'dartsmodel',
    'augmentcnn',
    'convnet4small'
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'convnet4small': convnet4,
    'resnet12': resnet12,
    'seresnet12': seresnet12,
    'resnet50': resnet50,
    'dartsmodel': NetworkMiniImageNet,
    'augmentcnn': AugmentCNN
}
