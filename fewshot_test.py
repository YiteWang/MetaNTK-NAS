from __future__ import print_function

import os
import argparse
import time
import sys

from pathlib import Path
lib_dir = (Path(__file__).parent / 'eval_lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import logging
import random
import numpy as np

from rfs_models import model_dict, model_pool
from rfs_models.util import create_model, count_params

from rfs_dataset.mini_imagenet import ImageNet, MetaImageNet
from rfs_dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from rfs_dataset.cifar import CIFAR100, MetaCIFAR100
from rfs_dataset.transform_cfg import transforms_options, transforms_list

from eval.meta_eval import meta_test
from ptflops import get_model_complexity_info

def parse_option(argv):

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='augmentcnn', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=50, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='~/data', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    # specify architectures for DARTS space
    parser.add_argument('--layers', type=int, default=4, help='number of layers')
    parser.add_argument('--init_channels', type=int, default=28, help='num of init channels')
    parser.add_argument('--genotype', type=str, default='', help='Cell genotype')
    parser.add_argument('--aug_stemm', type=int, default=3, help='Stem multiplier during augmentation')
    parser.add_argument('--aug_fsr', type=int, default=1, help='Feature scaling ratio during augmentation')
    
    # Parameters for Logistic regression
    parser.add_argument('--C', type=float, default=1.0, help='coefficient of Logistic Regression')
    parser.add_argument('--nonorm', action='store_false', dest='norm', help='if normalize feature, default: True')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    opt = parser.parse_args(argv)

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    opt.data_root = os.path.join(opt.data_root,opt.dataset)
    opt.data_aug = True

    return opt

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def main(argv):

    opt = parse_option(argv)

    if __name__ == '__main__':
        logging.basicConfig(filename= "rfs_results_{}shots.log".format(opt.n_shots),
                        format='%(asctime)s %(message)s',
                        filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger(__name__)

    prepare_seed(opt.seed)
    
    # test loader
    # args = opt
    opt.batch_size = opt.test_batch_size
    # args.n_aug_support_samples = 1

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    support_xs, _, _, _ = next(iter(meta_testloader))
    batch_size, _, channel, height, width = support_xs.size()

    # Get input channel/size for creating augmentcnn model
    if opt.model == 'augmentcnn':
        assert height == width
        opt.n_input_channels = channel
        opt.input_size = height

    # load model
    model = create_model(opt.model, n_cls, opt.dataset, args=opt )
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # Calculate model size & number of flops
    logger.info('Number of parameters: {}'.format(count_params(model)))

    macs, params = get_model_complexity_info(model, (channel, height, width), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)

    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    logger.info('Use norm:{}'.format(opt.norm))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False, is_norm=opt.norm, C=opt.C)
    test_time = time.time() - start
    logger.info('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main(sys.argv[1:])
