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
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n, get_analytical_metantk_n
from utils import get_model_infos
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
from nas_201_api import NASBench201API as API
from pdb import set_trace as bp
import eval_rfs
from opacus.utils import module_modification

INF = 1000  # used to mark prunned operators


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)

def convert_model(network, norm_type='groupnorm'):
    # Convert models with different normalization layers
    if norm_type == 'groupnorm':
        network_gn = module_modification.convert_batchnorm_modules(network, module_modification._batchnorm_to_groupnorm)
    elif norm_type == 'groupnorm16':
        def _batchnorm_to_groupnorm16(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
            return nn.GroupNorm(min(16, module.num_features), module.num_features, affine=True)
        network_gn = module_modification.convert_batchnorm_modules(network, _batchnorm_to_groupnorm16)
    elif norm_type == 'instancenorm':
        network_gn = module_modification.convert_batchnorm_modules(network, module_modification._batchnorm_to_instancenorm)
    elif norm_type == 'nonorm':
        network_gn = module_modification.nullify_batchnorm_modules(network)
    else:
        raise NotImplementedError('Opacus only support group/instance norm, please try to use nonorm to remove all norm layers')
    return network_gn.cuda().train()

def prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, logger, precision=10, prune_number=1):
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    if xargs.ntk_type in ['NTK','MetaNTK_anl']:
        network_origin = convert_model(network_origin, xargs.norm_type)
    init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_edge in range(len(arch_parameters[idx_ct])):
            # edge
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                # only one op remaining
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                # op
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    # this edge-op not pruned yet
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF
                    # ##### get ntk (score) ########
                    network = get_cell_based_tiny_net(model_config).cuda().train()
                    if xargs.ntk_type in ['NTK','MetaNTK_anl']:
                        network = convert_model(network, xargs.norm_type)
                    network.set_alphas(_arch_param)
                    ntk_delta = []
                    repeat = xargs.repeat
                    for _ in range(repeat):
                        # random reinit
                        init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                        # make sure network_origin and network are identical
                        for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                            param.data.copy_(param_ori.data)
                        network.set_alphas(_arch_param)
                        if xargs.only_lrs:
                            # If only use number of linear regions to prune, then generate same ntk cond values
                            ntk_origin, ntk = 1.0, 1.0
                        else:
                            if xargs.ntk_type == 'NTK':
                                ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=xargs.bn_mode, num_batch=1)
                            elif xargs.ntk_type == 'MetaNTK_anl':
                                ntk_origin, ntk = get_analytical_metantk_n(loader, [network_origin, network], xargs.n_ways, algorithm =xargs.algorithm, 
                                                                        inner_lr_time = xargs.inner_lr_time, reg_coef = xargs.reg_coef, recalbn=0, train_mode=xargs.bn_mode, 
                                                                        num_batch=1, params_types=xargs.params_types, norm_type=xargs.norm_type)
                        # ####################
                        ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))  # higher the more likely to be prunned
                    ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                    network.zero_grad()
                    network_origin.zero_grad()
                    #############################
                    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin_origin.set_alphas(arch_parameters)
                    network_thin_origin.train()
                    network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin.set_alphas(_arch_param)
                    network_thin.train()
                    with torch.no_grad():
                        _linear_regions = []
                        repeat = xargs.repeat
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                            # make sure network_thin and network_thin_origin are identical
                            for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                param.data.copy_(param_ori.data)
                            network_thin.set_alphas(_arch_param)
                            #####
                            lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                            _lr, _lr_2 = lrc_model.forward_batch_sample()
                            _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions, lower the more likely to be prunned
                            lrc_model.clear()
                        linear_regions = np.mean(_linear_regions)
                        regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                        choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                    #############################
                    torch.cuda.empty_cache()
                    del network_thin
                    del network_thin_origin
                    pbar.update(1)
    ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
    # print("NTK conds:", ntk_all)
    # logger.log('NTK conds max: {}'.format(ntk_all[0][0]))
    # logger.log('NTK conds min: {}'.format(ntk_all[-1][0]))
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
    for idx, data in enumerate(ntk_all):
        if idx == 0:
            rankings[data[1]] = [idx]
        else:
            if data[0] == ntk_all[idx-1][0]:
                # same ntk as previous
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
            else:
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
    # print(rankings)
    regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
    # print("#Regions:", regions_all)
    for idx, data in enumerate(regions_all):
        if idx == 0:
            rankings[data[1]].append(idx)
        else:
            if data[0] == regions_all[idx-1][0]:
                # same #Regions as previous
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
            else:
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    # ascending by sum of two rankings
    rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank] in rankings_sum:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
    choices_edges = list(edge2choice.values())
    # print("Final Ranking:", rankings_sum)
    # print("Pruning Choices:", choices_edges)
    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges


def prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, logger, edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2, precision=10):
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    if xargs.ntk_type in ['NTK','MetaNTK_anl']:
        network_origin = convert_model(network_origin, xargs.norm_type)
    init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    assert edge_groups[-1][1] == len(arch_parameters[0])
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_group in range(len(edge_groups)):
            edge_group = edge_groups[idx_group]
            # print("Pruning cell %s group %s.........."%("normal" if idx_ct == 0 else "reduction", str(edge_group)))
            if edge_group[1] - edge_group[0] <= num_per_group:
                # this group already meets the num_per_group requirement
                pbar.update(1)
                continue
            for idx_edge in range(edge_group[0], edge_group[1]):
                # edge
                for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                    # op
                    if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                        # this edge-op not pruned yet
                        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                        _arch_param[idx_ct][idx_edge, idx_op] = -INF
                        # ##### get ntk (score) ########
                        network = get_cell_based_tiny_net(model_config).cuda().train()
                        if xargs.ntk_type in ['NTK','MetaNTK_anl']:
                            network = convert_model(network, xargs.norm_type)
                        network.set_alphas(_arch_param)
                        ntk_delta = []
                        repeat = xargs.repeat
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                            # make sure network_origin and network are identical
                            for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                                param.data.copy_(param_ori.data)
                            network.set_alphas(_arch_param)
                            if xargs.only_lrs:
                                ntk_origin, ntk = 1.0, 1.0
                            else:
                                if xargs.ntk_type == 'NTK':
                                    ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=xargs.bn_mode, num_batch=1)
                                elif xargs.ntk_type == 'MetaNTK_anl':
                                    ntk_origin, ntk = get_analytical_metantk_n(loader, [network_origin, network], xargs.n_ways, algorithm =xargs.algorithm, 
                                                                                inner_lr_time = xargs.inner_lr_time, reg_coef = xargs.reg_coef, recalbn=0, train_mode=xargs.bn_mode, 
                                                                                num_batch=1, params_types=xargs.params_types, norm_type=xargs.norm_type)
                            # ####################
                            ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))
                        ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                        network.zero_grad()
                        network_origin.zero_grad()
                        #############################
                        network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin_origin.set_alphas(arch_parameters)
                        network_thin_origin.train()
                        network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin.set_alphas(_arch_param)
                        network_thin.train()
                        with torch.no_grad():
                            _linear_regions = []
                            repeat = xargs.repeat
                            for _ in range(repeat):
                                # random reinit
                                init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                                # make sure network_thin and network_thin_origin are identical
                                for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                    param.data.copy_(param_ori.data)
                                network_thin.set_alphas(_arch_param)
                                #####
                                lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                                _lr, _lr_2 = lrc_model.forward_batch_sample()
                                _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions
                                lrc_model.clear()
                            linear_regions = np.mean(_linear_regions)
                            regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                            choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                        #############################
                        torch.cuda.empty_cache()
                        del network_thin
                        del network_thin_origin
                        pbar.update(1)
            # stop and prune this edge group
            ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
            # print("NTK conds:", ntk_all)
            rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
            for idx, data in enumerate(ntk_all):
                if idx == 0:
                    rankings[data[1]] = [idx]
                else:
                    if data[0] == ntk_all[idx-1][0]:
                        # same ntk as previous
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
                    else:
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
            regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
            # print("#Regions:", regions_all)
            for idx, data in enumerate(regions_all):
                if idx == 0:
                    rankings[data[1]].append(idx)
                else:
                    if data[0] == regions_all[idx-1][0]:
                        # same #Regions as previous
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
                    else:
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
            rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            # ascending by sum of two rankings
            rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            choices = [item[0] for item in rankings_sum[:-num_per_group]]
            # print("Final Ranking:", rankings_sum)
            # print("Pruning Choices:", choices)
            for (cell_idx, edge_idx, op_idx) in choices:
                arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
            # reinit
            ntk_all = []  # (ntk, (edge_idx, op_idx))
            regions_all = []  # (regions, (edge_idx, op_idx))
            choice2regions = {}  # (edge_idx, op_idx): regions

    return arch_parameters


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    train_data, valid_data, xshape, class_num = get_datasets(xargs, -1)

    ##### config & logging #####
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = xargs.batch_size
    xargs.save_dir = xargs.save_dir + \
        "/repeat%d-prunNum%d-prec%d-%s-batch%d"%(
                xargs.repeat, xargs.prune_number, xargs.precision, xargs.init, config["batch_size"]) + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    config.save_dir = xargs.save_dir
    logger = prepare_logger(xargs)
    ###############

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'ntk_type': xargs.ntk_type,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                   'ntk_type': xargs.ntk_type,
                                  })
    elif xargs.search_space_name in ['darts', 'darts_fewshot']:
        # Different than TE-NAS, we do not use thin proxy network to compute NTK/MetaNTK,
        # Instead, we use network during augmentation to compute NTK/MetaNTK
        model_config = edict({'name': 'DARTS-V1',
                              # 'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'C': xargs.ntk_channels, 'N': xargs.ntk_N, 'depth': xargs.ntk_layers, 'use_stem': True, 'stem_multiplier': xargs.ntk_stemm,
                              'feature_scale_rate': xargs.ntk_fsr,
                              'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'super_type': xargs.super_type,
                              'steps': xargs.max_nodes,
                              'multiplier': xargs.max_nodes,
                              'ntk_type': xargs.ntk_type,
                             })
        # For linear regions, we keep the same method used in TE-NAS
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                   'feature_scale_rate': 2,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                   'super_type': xargs.super_type,
                                   'steps': xargs.max_nodes,
                                   'multiplier': xargs.max_nodes,
                                   'ntk_type': xargs.ntk_type,
                                  })
    network = get_cell_based_tiny_net(model_config)
    logger.log('model-config : {:}'.format(model_config))
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0

    # TODO Linear_Region_Collector
    lrc_model = Linear_Region_Collector(xargs, input_size=(1000, 1, 3, 3), sample_batch=3, dataset=xargs.dataset, data_path=xargs.data_path, seed=xargs.rand_seed)

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None or xargs.search_space_name in ['darts', 'darts_fewshot']:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    network = network.cuda()

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
    arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
    np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-6:])))

        arch_parameters, op_pruned = prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space, logger,
                                                     precision=xargs.precision,
                                                     prune_number=xargs.prune_number
                                                     )
        # rebuild supernet
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)

        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
        genotypes['arch'][epoch] = network.genotype()

        logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    if xargs.search_space_name in ['darts', 'darts_fewshot']:
        print("===>>> Prune Edge Groups...")
        if xargs.max_nodes == 4:
            edge_groups = [(0, 2), (2, 5), (5, 9), (9, 14)]
        elif xargs.max_nodes == 3:
            edge_groups = [(0, 2), (2, 5), (5, 9)]
        arch_parameters = prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space, logger,
                                                edge_groups=edge_groups , num_per_group=2,
                                                precision=xargs.precision,
                                                )
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
    else:
        raise NotImplementedError('Only support darts and darts_fewshot search space.')

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-'*100)
    logger.log("Time spent: %d s"%(end_time - start_time))

    # Delete existing model and data loader
    searched_arch = str(network.genotype())
    del network
    del train_loader
    del train_data
    del valid_data

    # Augmenting Architecture searched by our search method if train_after_search is True
    if  xargs.train_after_search:
        if xargs.dataset in ['MiniImageNet', 'MetaMiniImageNet']:
            dataset_for_eval = 'miniImageNet'
        elif xargs.dataset in ['TieredImageNet', 'MetaTieredImageNet']:
            dataset_for_eval = 'tieredImageNet'
        else:
            raise NotImplementedError('Only support train after search for miniImageNet and tieredImageNet')

        if xargs.train_method == 'rfs':
            evaluation_cmd = ['--model', 'augmentcnn', 
                    '--dataset', dataset_for_eval, '--data_root', os.path.dirname(xargs.data_path),
                    '--init_channels', str(xargs.aug_channels), '--layers', str(xargs.aug_layers),  '--aug_dp', str(xargs.aug_dp),
                    '--aug_stemm', str(xargs.aug_stemm), '--aug_fsr', str(xargs.aug_fsr),
                    '--lr_decay_epochs', str(xargs.aug_lr_decay_epochs), '--epochs', str(xargs.aug_epochs),
                    '--learning_rate', str(xargs.aug_lr),
                    '--seed', '-1',
                    '--batch_size', str(xargs.aug_batchsize), '--genotype', searched_arch, 
                    '--tb_path', os.path.join(str(xargs.save_dir),'logs'), '--model_path', os.path.join(str(xargs.save_dir),'model'),]
            # 5 way 5 shot
            eval_rfs.main(evaluation_cmd)
        else:
            raise NotImplementedError('Only support rfs training now.')

    # check the performance from the architecture dataset
    if api is not None:
        logger.log('{:}'.format(api.query_by_arch(genotypes['arch'][epoch])))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MetaNTK_NAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['MiniImageNet', 'MetaMiniImageNet', 'TieredImageNet', 'MetaTieredImageNet'], help='Choose which dataset to search')
    parser.add_argument('--search_space_name', type=str, default='darts_fewshot', choices=['darts', 'darts_fewshot'], help='space of operator candidates: darts or darts_fewshot.')
    parser.add_argument('--max_nodes', type=int, choices=[3,4], default=3, help='The maximum number of nodes, choose from 3 and 4')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], default=1, help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for NTK/MetaNTK')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed, set to -1 for random seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')
    parser.add_argument('--super_type', type=str, default='nasnet-super',  help='type of supernet: basic or nasnet-super')

    # Setting for meta data loader
    parser.add_argument('--transform', type=str, default='TENAS', choices = ['A', 'B', 'C', 'D', 'TENAS'], help = 'Transformation of datasets for NTK computation')
    parser.add_argument('--data_aug', type=bool, default=True)
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N', help='Length of the meta dataloader for arch search')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of support samples of the meta dataloader for arch search')
    parser.add_argument('--n_queries', type=int, default=1, metavar='N', help='Number of query samples of the meta dataloader for arch search')
    parser.add_argument('--n_aug_support_samples', default=1, type=int, help='The number of augmented samples of the meta dataloader for arch search')
    
    # Choosing which kind of NTK to compute
    parser.add_argument('--ntk_type', type=str, default='MetaNTK_anl', choices = ['NTK', 'MetaNTK_anl'], help='Compute NTK or MetaNTK_analytical for searching')
    
    # Search architecture setting for computing NTK/MetaNTK
    parser.add_argument('--ntk_channels', type=int, default=48, help='initial channels of network for computing NTKs. To use Opacus for analytical MetaNTK, use 16n channels.')
    parser.add_argument('--ntk_layers', type=int, default=5, help='number of layers of network for computing NTKs')
    parser.add_argument('--ntk_N', type=int, default=-1, help='number of layers for each stage, default (ntk_layers-2)//3')
    parser.add_argument('--ntk_stemm', type=int, default=1, help='Stem multiplier when computing NTK')
    parser.add_argument('--ntk_fsr', type=int, default=2, help='Feature scaling ratio when computing NTK')
    parser.add_argument('--only_lrs', choices = ['true', 'false'], default='false', help='Use only linear regions')

    # Arguments for computing analytical MetaNTK
    parser.add_argument('--algorithm', type=str, default='MAML', choices = ['ANIL', 'MAML'], help='Algorithm for computing analytical MetaNTK')
    parser.add_argument('--inner_lr_time', type=float, default=1000.0, help='the product of inner loop learning rate & training time')
    parser.add_argument('--reg_coef', type=float, default=1e-3, help='the regularization coefficient for the inner loop optimization. suggest >=1e-5')
    parser.add_argument('--bn_mode', type=str, choices = ['train', 'eval'], default='train', help='Mode of batchnorm when computing NTK/MetaNTK, default: train mode')
    parser.add_argument('--params_types', type=str, choices = ['wb', 'w'], default='w', help='What params to compute analytical NTK, wb stands for weight+bias, w stands for weight')
    parser.add_argument('--norm_type', type=str, choices = ['nonorm', 'groupnorm', 'instancenorm', 'groupnorm16'], 
        default='groupnorm16', help='Which norm to replace batchnorm, nonorm to replace batchnorm with identity')

    # Settings for training after search for RFS
    parser.add_argument('--train_after_search', choices = ['true', 'false'], default='false', help='If directly train after search with RFS.')
    parser.add_argument('--train_method', default='rfs', choices = ['rfs'], help='What evaluation method used to train the architecture')
    parser.add_argument('--aug_dp', type=float, default=0.2, help='Drop probability of augmentCNN')
    parser.add_argument('--aug_channels', type=int, default=48, help='Init channels for network during augmentation')
    parser.add_argument('--aug_layers', type=int, default=5, help='Number of layers for network during augmentation')
    parser.add_argument('--aug_lr', type=float, default=0.02, help='Learning rate for network during augmentation')
    parser.add_argument('--aug_batchsize', type=int, default=64, help='Batch size for network during augmentation')
    parser.add_argument('--aug_epochs', type=int, default=100, help='Total number of epochs for network during augmentation')
    parser.add_argument('--aug_lr_decay_epochs', type=str, default='60,80', help='Learning rate decay epochs during augmentation')
    parser.add_argument('--aug_stemm', type=int, default=1, help='Stem multiplier during augmentation')
    parser.add_argument('--aug_fsr', type=int, default=2, help='Feature scaling ratio during augmenation')

    args = parser.parse_args()

    if args.ntk_N == -1:
        args.ntk_N = int((1.0 * args.ntk_layers - 2)/3)
    else:
        if not (args.ntk_N * 3 + 2 == args.ntk_layers):
            print('[*] You may want to check NTK architecture again, N*3+2 is not equal to ntk_layers.')

    if (args.ntk_stemm!=args.aug_stemm) or (args.ntk_fsr!=args.aug_fsr) or (args.ntk_layers!=args.aug_layers) or (args.ntk_channels!=args.aug_channels):
        print('[*] Note that architectures during NTK computation and augmenation are different.')

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    # If use MetaNTK, assert that dataset is using Meta version of mini/tieredImageNet
    if args.ntk_type == 'MetaNTK_anl':
        assert args.dataset in ['MetaMiniImageNet', 'MetaTieredImageNet']

    # Doesn't matter if using groupnorm/groupnorm16/instancenorm/layernorm
    if args.bn_mode == 'train':
        args.bn_mode = True
    else:
        args.bn_mode = False

    # If search using solely number of linear regions
    if args.only_lrs == 'true':
        args.only_lrs = True
    else:
        args.only_lrs = False

    # If augment searched architecture after search is done
    if args.train_after_search == 'true':
        args.train_after_search = True
    else:
        args.train_after_search = False

    # For inner_lr_time larger than 1000, we set it to be Inf
    if args.inner_lr_time >= 1000:
        args.inner_lr_time = np.inf
    main(args)
