from __future__ import print_function

import os
import argparse
import socket
import time
import sys

from pathlib import Path
lib_dir = (Path(__file__).parent / 'eval_lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import random
import numpy as np

import logging

from rfs_models import model_pool
from rfs_models.util import create_model, count_params

from rfs_dataset.mini_imagenet import ImageNet, MetaImageNet
from rfs_dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from rfs_dataset.cifar import CIFAR100, MetaCIFAR100
from rfs_dataset.transform_cfg import transforms_options, transforms_list

from rfs_util import adjust_learning_rate, accuracy, AverageMeter
from flop_benchmark import get_model_infos

from eval.meta_eval import meta_test
from eval.cls_eval import validate

from ptflops import get_model_complexity_info

import torch_optimizer
import fewshot_test

def parse_option(argv):

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--resume', default='', type=str, help='Checkpoint path for resume training.')
    parser.add_argument('--start_epoch', type=int, default=1, help='Start epoch')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--radam', action='store_true', help='use Radam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='default', help='using cosine annealing', choices=['default', 'cosine', 'reducelronplateau','steplr'])

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    parser.add_argument('--seed', type=int, default=-1, help='Random seed')

    # specify architectures for DARTS space
    parser.add_argument('--layers', type=int, default=5, help='number of layers')  
    parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
    parser.add_argument('--genotype', type=str, default='', help='Cell genotype')
    parser.add_argument('--aug_stemm', type=int, default=1, help='Stem multiplier during augmentation')
    parser.add_argument('--aug_fsr', type=int, default=2, help='Feature scaling ratio during augmentation')
    parser.add_argument('--aug_dp', type=float, default=0.2, help='Drop probability of augmentCNN')

    opt = parser.parse_args(argv)

    if opt.seed is None or opt.seed < 0:
        opt.seed = random.randint(1, 100000)

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './logger'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if not os.path.isdir(opt.tb_path):
        os.makedirs(opt.tb_path)

    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)

    opt.n_gpu = torch.cuda.device_count()

    return opt

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def main(argv):

    #Control randomness during backbone training
    opt = parse_option(argv)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    prepare_seed(opt.seed)

    # Create directory if not exist
    Path(opt.tb_path).mkdir(parents=True, exist_ok=True)

    # Create and configure logger
    logging.basicConfig(filename= os.path.join(opt.tb_path,"rfs_results.log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
      
    # Creating a logger object
    logger=logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Print all arguments
    logger.info('All arguments: \n')
    for arg in vars(opt):
        logger.info('{}: {}'.format(arg, getattr(opt, arg)))

    # Create dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
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

    # Get input channel/size for creating augmentcnn model
    support_xs, _, _, _ = next(iter(meta_testloader))
    batch_size, _, channel, height, width = support_xs.size()

    if opt.model == 'augmentcnn':
        assert height == width
        opt.n_input_channels = channel
        opt.input_size = height
 
    # Create model
    model = create_model(opt.model, n_cls, opt.dataset, args=opt )

    # Calculate model size & number of flops
    logger.info('Number of parameters: {}'.format(count_params(model)))

    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)
    elif opt.radam:
        optimizer = torch_optimizer.RAdam(model.parameters(),
                            lr = opt.learning_rate,
                            weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    logger.info('{:<30}  {}'.format('Shape of input: ', (channel, height, width)))

    macs, params = get_model_complexity_info(model, (channel, height, width), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)

    logger.info('Results by PTFLOPS:')
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    flop, param = get_model_infos(model, (1, channel, height, width))
    logger.info('Results by TE-NAS:')
    logger.info('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))

    # set learning rate scheduler
    if opt.scheduler == 'cosine':
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)
    elif opt.scheduler == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif opt.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)

    # Resume from checkpoint
    if opt.resume:
        assert os.path.isfile(opt.resume), 'Unable to find checkpoint file.'
        print('Loading checkpoint : {}'.format(opt.resume))
        checkpoint = torch.load(opt.resume)
        assert opt.genotype == checkpoint['genotype'], 'Genotype mismatch, saved genotype: {}'.format(checkpoint['genotype'])
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        if opt.scheduler != 'default':
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info('Load checkpoint from {}, start from epoch {}'.format(opt.resume, checkpoint['epoch']))


    best_acc = 0
    best_epoch = 0
    
    # routine: supervised pre-training
    for epoch in range(opt.start_epoch, opt.epochs + 1):

        # Gradually change drop_path_prob as DARTS did
        if opt.aug_dp != 0.0:
            assert opt.model == 'augmentcnn', 'aug_dp is only used for AugmentCNN'
            model.drop_path_prob(opt.aug_dp * epoch / opt.epochs)

        if opt.scheduler == 'default':
            adjust_learning_rate(epoch, opt, optimizer)

        logger.info("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.info('train_acc at epoch {}: {}'.format(epoch, train_acc))
        logger.info('train_loss at epoch {}: {}'.format(epoch, train_loss))

        # regular saving
        logger.info('==> Saving...')
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            'opt': optimizer.state_dict(),
            'scheduler': None if opt.scheduler == 'default' else scheduler.state_dict(),
            'genotype': opt.genotype,
        }
        save_file = os.path.join(opt.model_path, 'ckpt_last.pth')
        torch.save(state, save_file)

        ## You may also want to save best checkpoint during training.

        # if epoch % opt.save_freq == 0:
        #     # Instead of using direct validation, we use validation with fine-tuning
        #     # test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        #     test_acc, _  = meta_test(model, meta_valloader, use_logit=False)

        #     logger.info('val_acc with feature at epoch {}: {}'.format(epoch, test_acc))
        #     # logger.info('test_acc_top5 at epoch {}: {}'.format(epoch, test_acc_top5))
        #     # logger.info('test_loss at epoch {}: {}'.format(epoch, test_loss))

        #     if test_acc > best_acc:
        #         best_acc = test_acc
        #         best_epoch = epoch
        #         logger.info('==> Saving best checkpoint...')
        #         state = {
        #             'epoch': epoch + 1,
        #             'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        #             'opt': optimizer.state_dict(),
        #             'scheduler': None if opt.scheduler == 'default' else scheduler.state_dict(),
        #             'genotype': opt.genotype,
        #         }
        #         save_file = os.path.join(opt.model_path, 'ckpt_best.pth')
        #         torch.save(state, save_file)


        # Save checkpoint for each epoch after 2/3 number of epochs of training
        if epoch >= int(1.0*opt.epochs*2/3):
            save_file = os.path.join(opt.model_path, 'ckpt_epoch{}.pth'.format(epoch))
            torch.save(state, save_file)

        if opt.scheduler == 'reducelronplateau':
            scheduler.step(test_acc)
        elif opt.scheduler in ['cosine','steplr']:
            scheduler.step()

    # save the last model
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        'opt': optimizer.state_dict(),
        'scheduler': None if opt.scheduler == 'default' else scheduler.state_dict(),
        'genotype': opt.genotype,
    }
    save_file = os.path.join(opt.model_path, 'ckpt_final.pth'.format(opt.model))
    torch.save(state, save_file)
    logger.info('Best Acc is at epoch {} with accuracy: {} '.format(best_epoch, best_acc))

    logger.info('Start to few-shot test:')
    logger.info('1 shot result:')
    cmd_1shot = ['--model', opt.model, 
                    '--dataset', opt.dataset, '--data_root', os.path.dirname(opt.data_root),
                    '--init_channels', str(opt.init_channels), '--layers', str(opt.layers), 
                    '--aug_stemm', str(opt.aug_stemm), '--aug_fsr', str(opt.aug_fsr),
                    '--genotype', opt.genotype,
                    # '--model_path', os.path.join(opt.model_path, 'ckpt_best.pth'),
                    '--model_path', os.path.join(opt.model_path, 'ckpt_final.pth'),
                    '--n_shots', '1']
    fewshot_test.main(cmd_1shot)

    logger.info('5 shots result:')
    cmd_5shot = ['--model', opt.model, 
                    '--dataset', opt.dataset, '--data_root', os.path.dirname(opt.data_root),
                    '--init_channels', str(opt.init_channels), '--layers', str(opt.layers), 
                    '--aug_stemm', str(opt.aug_stemm), '--aug_fsr', str(opt.aug_fsr),
                    '--genotype', opt.genotype,
                    # '--model_path', os.path.join(opt.model_path, 'ckpt_best.pth'),
                    '--model_path', os.path.join(opt.model_path, 'ckpt_final.pth'),
                    '--n_shots', '5']
    fewshot_test.main(cmd_5shot)

def train(epoch, train_loader, model, criterion, optimizer, opt):
    logger = logging.getLogger(__name__)
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main(sys.argv[1:])
