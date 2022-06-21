import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from procedures import prepare_seed, prepare_logger
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
import eval_rfs

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def main(xargs):
    PID = os.getpid()
    prepare_seed(xargs.rand_seed)
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    ##### logging #####
    xargs.save_dir = xargs.save_dir + \
        "Randombaseline" + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    logger = prepare_logger(xargs)
    ###############

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': xargs.max_nodes, 'num_classes': 1,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'ntk_type': 'NTK',
                             })

    elif xargs.search_space_name in ['darts', 'darts_fewshot']:
        model_config = edict({'name': 'DARTS-V1',
                      'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                      'feature_scale_rate': 2,
                      'num_classes': 1,
                      'space': search_space,
                      'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                      'super_type': xargs.super_type,
                      'steps': xargs.max_nodes,
                      'multiplier': xargs.max_nodes,
                      'ntk_type': 'NTK',
                     })

    network = get_cell_based_tiny_net(model_config)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))

    if xargs.dataset in ['MiniImageNet', 'MetaMiniImageNet']:
        dataset_for_eval = 'miniImageNet'
    elif xargs.dataset in ['TieredImageNet', 'MetaTieredImageNet']:
        dataset_for_eval = 'tieredImageNet'
    else:
        raise NotImplementedError('Only support miniImageNet and tieredImageNet')

    if xargs.train_method == 'rfs':
        evaluation_cmd = ['--model', 'augmentcnn', 
                '--dataset', dataset_for_eval, '--data_root', os.path.dirname(xargs.data_path),
                '--init_channels', str(xargs.aug_channels), '--layers', str(xargs.aug_layers),  '--aug_dp', str(xargs.aug_dp),
                '--aug_stemm', str(xargs.aug_stemm), '--aug_fsr', str(xargs.aug_fsr),
                '--lr_decay_epochs', str(xargs.aug_lr_decay_epochs), '--epochs', str(xargs.aug_epochs),
                '--learning_rate', str(xargs.aug_lr),
                '--seed', '-1',
                '--batch_size', str(xargs.aug_batchsize), '--genotype', str(network.genotype()), 
                '--tb_path', os.path.join(str(xargs.save_dir),'logs'), '--model_path', os.path.join(str(xargs.save_dir),'model'),]
        # 5 way 5 shot
        eval_rfs.main(evaluation_cmd)
    else:
        raise NotImplementedError('Only support rfs training now.')

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random_baseline")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['MiniImageNet', 'MetaMiniImageNet', 'TieredImageNet', 'MetaTieredImageNet'], help='Choose dataset')
    parser.add_argument('--search_space_name', type=str, default='darts_fewshot',  help='space of operator candidates: nas-bench-201 or darts or darts_fewshot.')
    parser.add_argument('--max_nodes', type=int, choices=[3,4], default=3, help='The maximum number of nodes, choose from 3 and 4')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--super_type', type=str, default='nasnet-super',  help='type of supernet: basic or nasnet-super')
    parser.add_argument('--rand_seed', type=int, help='manual seed')

    # Train after search for RFS
    parser.add_argument('--train_after_search', choices = ['true', 'false'], default='true', help='If directly train after search with RFS')
    parser.add_argument('--train_method', default='rfs', choices = ['rfs'], help='What evaluation method used to train the architecture')
    parser.add_argument('--aug_dp', type=float, default=0.2, help='Drop probability of augmentCNN')
    parser.add_argument('--aug_channels', type=int, default=48, help='Init channels for network during augmentation')
    parser.add_argument('--aug_layers', type=int, default=5, help='Number of layers for network during augmentation')
    parser.add_argument('--aug_lr', type=float, default=0.02, help='Learning rate for network during augmentation')
    parser.add_argument('--aug_batchsize', type=int, default=64, help='Batch size for network during augmentation')
    parser.add_argument('--aug_epochs', type=int, default=100, help='Batch size for network during augmentation')
    parser.add_argument('--aug_lr_decay_epochs', type=str, default='60,80', help='Learning rate decay epochs during augmentation')
    parser.add_argument('--aug_stemm', type=int, default=1, help='Stem multiplier during augmentation')
    parser.add_argument('--aug_fsr', type=int, default=2, help='Feature scaling ratio during augmenation')

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    main(args)
