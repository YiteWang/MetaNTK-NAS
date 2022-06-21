from __future__ import print_function


from . import model_dict
  
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def create_model(name, n_cls, dataset='miniImageNet', args=None):
    """create model by name"""
    if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls)
        elif name.startswith('wrn'):
            model = model_dict[name](num_classes=n_cls)
        elif name == 'convnet4small':
            model = model_dict[name](num_classes=n_cls, hidden_size=32)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('dartsmodel'):
            assert args is not None
            assert args.genotype != ''
            genotype = eval(args.genotype)
            model = model_dict[name](args, args.init_channels, n_cls, args.layers, criterion=None, auxiliary=None, genotype=genotype) 
        elif name == 'augmentcnn':
            assert args is not None
            assert args.genotype != ''
            genotype = eval(args.genotype)
            model = model_dict[name](input_size=args.input_size, C_in=args.n_input_channels, C=args.init_channels, n_classes=n_cls, n_layers=args.layers, auxiliary=False, genotype=genotype, stem_multiplier=args.aug_stemm, feature_scale_rate=args.aug_fsr)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CIFAR-FS' or dataset == 'FC100':
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    return model

def count_params(net):
    return sum(p.numel() for p in net.parameters())
    
def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
