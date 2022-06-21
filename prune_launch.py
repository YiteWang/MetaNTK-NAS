import os
import time
import argparse

# TODO please configure data_paths before running, please leave TORCH_HOME empty
TORCH_HOME = ""  
data_paths = {
    "MiniImageNet": "~/data/miniImageNet",
    "TieredImageNet": "~/data/tieredImageNet",
    "MetaMiniImageNet": "~/data/miniImageNet",
    "MetaTieredImageNet": "~/data/tieredImageNet",
}


parser = argparse.ArgumentParser("MetaNTK_NAS_launch")
parser.add_argument('--gpu', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--space', default='darts_fewshot', type=str, choices=['darts', 'darts_fewshot'], help='which nas search space to use')
parser.add_argument('--dataset', default='MiniImageNet', type=str, choices=['MiniImageNet', 'TieredImageNet', 'MetaMiniImageNet', 'MetaTieredImageNet'], help='Choose from MiniImageNet/TieredImageNet')
parser.add_argument('--seed', default=-1, type=int, help='manual seed, set to -1 for random seed')
parser.add_argument('--max_nodes', default=3, type=int, help='number of max nodes, 4 for darts and 3 for darts_fewshot')
parser.add_argument('--dartsbs', default=3, type=int, help = 'Batch size of NTK/MetaNTK when using darts or darts_fewshot search space on imagenet subset (mini, imagenet-1k), default 24.')
parser.add_argument('--ntk_type', default='MetaNTK_anl', type=str, choices = ['NTK', 'MetaNTK_anl'], help = 'To compute NTK or MetaNtk')
parser.add_argument('--ntk_channels', type=int, default=48, help='initial channels of small network for computing NTKs. To use Opacus, use 16n channels.')
parser.add_argument('--ntk_layers', type=int, default=5, help='number of layers of small network for computing NTKs')
parser.add_argument('--only_lrs', choices = ['true', 'false'], default='false', help='Use only linear regions')

# Arguments for computing analytical MetaNTK
parser.add_argument('--algorithm', type=str, default='MAML', choices = ['ANIL', 'MAML'], help='Algorithm for computing analytical MetaNTK')
parser.add_argument('--inner_lr_time', type=float, default=1000.0, help='the product of inner loop learning rate & training time')
parser.add_argument('--reg_coef', type=float, default=1e-3, help='the regularization coefficient for the inner loop optimization. suggest >=1e-5')

# Train after search
parser.add_argument('--train_after_search', choices = ['true', 'false'], default='false', help='If directly train after search with RFS.')
parser.add_argument('--aug_dp', type=float, default=0.2, help='Drop probability of augmentCNN')
parser.add_argument('--aug_channels', type=int, default=48, help='Init channels for network during augmentation')
parser.add_argument('--aug_layers', type=int, default=5, help='Number of layers for network during augmentation')
parser.add_argument('--aug_lr', type=float, default=0.02, help='Learning rate for network during augmentation')
parser.add_argument('--aug_batchsize', type=int, default=64, help='Batch size for network during augmentation')
parser.add_argument('--aug_epochs', type=int, default=100, help='Total number of epochs for network during augmentation')
parser.add_argument('--aug_lr_decay_epochs', type=str, default='60,80', help='Learning rate decay epochs during augmentation')


args = parser.parse_args()

##### Basic Settings
precision = 3
# init = 'normal'
# init = 'kaiming_uniform'
init = 'kaiming_normal'

space = args.space
super_type = "nasnet-super"
batch_size = args.dartsbs

if args.ntk_type == 'MetaNTK_anl':
    assert args.dataset in ['MetaMiniImageNet', 'MetaTieredImageNet'], 'To use MetaNTK-NAS, please use meta version of the dataset.'

# ONLY TRAIN AFTER SEARCH FOR OUR SETTINGS
if args.train_after_search == 'true':
    assert (args.ntk_layers in [5,8]) and (args.ntk_channels == 48)
    args.aug_channels = args.ntk_channels
    args.aug_layers = args.ntk_layers
    if args.dataset in ['MiniImageNet', 'MetaMiniImageNet']:
        args.aug_dp = 0.2
        if args.ntk_layers == 8:
            args.aug_batchsize = 40 # Or change to 64 if your memory is big enough
        else:
            args.aug_batchsize = 64
        args.aug_lr = 0.02
        args.aug_epochs = 100
        args.aug_lr_decay_epochs = '60,80'
    elif args.dataset in ['TieredImageNet', 'MetaTieredImageNet']:
        args.aug_dp = 0.1
        if args.ntk_layers == 8:
            args.aug_batchsize = 56 # Or change to 64 if your memory is big enough
        else:
            args.aug_batchsize = 64
        args.aug_lr = 0.01
        args.aug_epochs = 60
        args.aug_lr_decay_epochs = '30,40,50'
    else:
        raise NotImplementedError

timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))


core_cmd = "CUDA_VISIBLE_DEVICES={gpuid} OMP_NUM_THREADS=4 python ./prune_metantknas.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--timestamp {timestamp} \
--precision {precision} \
--init {init} \
--repeat 3 \
--batch_size {batch_size} \
--ntk_type {ntk_type} \
--algorithm {algorithm} \
--inner_lr_time {inner_lr_time} \
--reg_coef {reg_coef} \
--train_after_search {train_after_search} \
--ntk_channels {ntk_channels} \
--ntk_layers {ntk_layers} \
--only_lrs {only_lrs} \
--aug_dp {aug_dp} \
--aug_channels {aug_channels} \
--aug_layers {aug_layers} \
--aug_lr {aug_lr} \
--aug_batchsize {aug_batchsize} \
--aug_epochs {aug_epochs} \
--aug_lr_decay_epochs {aug_lr_decay_epochs} \
".format(
    gpuid=args.gpu,
    save_dir="./output/prune-{space}/{dataset}".format(space=space, dataset=args.dataset),
    max_nodes=args.max_nodes,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=space,
    super_type=super_type,
    seed=args.seed,
    timestamp=timestamp,
    precision=precision,
    init=init,
    batch_size=batch_size,
    ntk_type=args.ntk_type,
    algorithm=args.algorithm,
    inner_lr_time=args.inner_lr_time,
    reg_coef=args.reg_coef,
    train_after_search=args.train_after_search,
    ntk_channels=args.ntk_channels,
    ntk_layers=args.ntk_layers,
    only_lrs=args.only_lrs,
    aug_dp=args.aug_dp,
    aug_channels=args.aug_channels,
    aug_layers=args.aug_layers,
    aug_lr=args.aug_lr,
    aug_batchsize=args.aug_batchsize,
    aug_epochs=args.aug_epochs,
    aug_lr_decay_epochs=args.aug_lr_decay_epochs,
)

os.system(core_cmd)
