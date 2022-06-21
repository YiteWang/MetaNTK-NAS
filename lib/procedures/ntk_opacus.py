import numpy as np
import torch
import torch.nn as nn
import scipy
import os
import opacus
import time
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector
from opacus.grad_sample import GradSampleModule

''' Opacus version of NTK/MetaNTK computations'''

NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LayerNorm,)

def recal_bn(network, xloader, recalbn, device):
    # Recalculate batchnorm statistics
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network

def kernel_T_func(kernel, time_evolution, reg_coef):
    '''
    The function kernel^{-1} (I - exp(-time_evolution* kernel))
    :param kernel: a symmetric PSD matrix
    :param time_evolution: the product of training time & learning rate
    :param reg_coef: the L2 regualrization coefficient ( we suggest using a small number instead of 0.
    e.g., >= 1e-5
    :return: (kernel + reg_coef* I) ^{-1} (I - exp(-time_evolution* kernel))
    '''
    I = np.eye(len(kernel))
    # pinvh is for symmetric matrix. Positive reg_coef can relieve the rank-deficiency issue
    kernel_inv = scipy.linalg.pinvh(kernel + reg_coef * I)
    # expm is matrix exponential function
    if time_evolution == np.inf:
        exp_term = 0
    else:
        exp_term = scipy.linalg.expm(-time_evolution * kernel)
    result = kernel_inv @ (I - exp_term)
    return result

def split_kernel_into_blocks(kernel, n_tasks):
    K = kernel
    K_vsplit = np.vsplit(K, n_tasks)
    blocks = [np.hsplit(sub_K, n_tasks) for sub_K in K_vsplit]
    return blocks

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def del_grad_sample(model):
    """
    Deletes ``.grad_sample`` from model's parameters. Code modified from:
    https://opacus.ai/api/_modules/opacus/grad_sample/grad_sample_module.html
    """
    for p in model.parameters():
        if hasattr(p, "grad_sample") and p.grad_sample is not None:
            if p.grad_sample.grad_fn is not None:
                p.grad_sample.detach_()
            else:
                p.grad_sample.requires_grad_(False)

            del p.grad_sample

def compute_analytical_metantk_n(ntk, K, n_tasks, n_queries, n_support, inner_lr_time, reg_coef, algorithm='ANIL'):
    '''
    Notice that the analytical metaNTK is derived for L2 loss only in the inf-width limit
    :param ntk: a symemtric torch tensor of with length = n_tasks*(n_queries+n_support).
        Notice that the first n_tasks*n_queries indices are for query samples and the last
        n_tasks*n_support indices are for support samples
    :param K: a symemtric torch tensor in the same shape as ntk. For MAML, K == ntk. For ANIL, K is NNGP
    :param n_tasks: the number of tasks in the NTK computation
    :param n_queries: number of query samples per task
    :param n_support: number of query samples per task
    :param inner_lr_time: the product of inner loop learning rate & training time
    :param reg_coef: the regularization coefficient for the inner loop optimization. suggest >=1e-5
    :param algorithm: choices = ['MAML','ANIL','MTL','iMAML']
    :return: the analytical metaNTK of shape
    '''

    ntk = ntk.detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    assert ntk.shape == K.shape

    assert len(ntk.shape) == 2
    assert ntk.shape[0] == ntk.shape[1]
    assert len(ntk) == n_tasks * (n_queries + n_support)  # check the dimension is correct
    assert check_symmetric(ntk) and check_symmetric(K)  # check the two matrices are both symmetric
    # we can support ANIL, MAML, Multi-Task-Learning, iMAML latter
    assert algorithm in ['ANIL', 'MAML', 'MTL']
    if algorithm == 'MAML':
        assert np.array_equal(ntk,K)
    elif algorithm == 'ANIL':
        assert not np.allclose(ntk,K)

    query_size = n_tasks * n_queries
    ntk_qry_blocks = split_kernel_into_blocks(ntk[:query_size, :query_size], n_tasks)
    ntk_spt_blocks = split_kernel_into_blocks(ntk[query_size:, query_size:], n_tasks)
    ntk_qry_spt_blocks = split_kernel_into_blocks(ntk[:query_size, query_size:], n_tasks)
    ntk_spt_qry_blocks = split_kernel_into_blocks(ntk[query_size:, :query_size], n_tasks)

    K_spt_blocks = split_kernel_into_blocks(K[query_size:, query_size:], n_tasks)
    K_qry_spt_blocks = split_kernel_into_blocks(K[:query_size, query_size:], n_tasks)

    KTs = []
    for i in range(n_tasks):
        T = kernel_T_func(kernel=K_spt_blocks[i][i], time_evolution=inner_lr_time,reg_coef=reg_coef)
        KT = np.matmul(K_qry_spt_blocks[i][i],  T) # matrix multiplication
        KTs.append(KT)

    nonsym_term = [[] for _ in range(n_tasks)]
    long_term = [[] for _ in range(n_tasks)]

    metantk_blocks = [[] for _ in range(n_tasks)]
    for i in range(n_tasks):
        for j in range(n_tasks):
            ntk_term = ntk_qry_blocks[i][j]
            KT_i = KTs[i]
            TK_j = np.transpose(KTs[j])
            KT_ntk_term = np.matmul(KT_i, ntk_spt_qry_blocks[i][j])
            ntk_TK_term = np.matmul(ntk_qry_spt_blocks[i][j], TK_j)
            KT_ntk_TK   = np.matmul(np.matmul(KT_i, ntk_spt_blocks[i][j]), TK_j)
            block = ntk_term - KT_ntk_term - ntk_TK_term + KT_ntk_TK
            metantk_blocks[i].append(block)

    metantk = np.block(metantk_blocks)
    assert check_symmetric(metantk)

    return {'metantk': metantk, 'eigenvalues': np.linalg.eigh(metantk)[0]}

def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1, params_types='w'):
    '''
    Here we compute NTK based on Opacus.
    :param xloader: A dataloader whose each batch contains images and corresponding labels
    :param networks: A list of networks to compute NTK condition numbers
    :param recalbn: If recalculate batchnorm statistics
    :param train_mode: If enable batchnorm statistics to update during computing NTK
    :param num_batch: Number of batches to average when computing NTK
    :return: NTK condition numbers whose size is the same as size of networks
    '''
    device = torch.cuda.current_device()

    if params_types == 'w':
        params_types = ['weight',]
    elif params_types == 'wb':
        params_types = ['weight', 'bias']

    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)

    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()

    ######

    ntks = []

    for i, datapair in enumerate(xloader):

        # Only need images to compute NTKs
        inputs = datapair[0]
        targets = datapair[1]
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)

        for net_idx, network in enumerate(networks):

            network = network.to(device)
            
            # Wrap network using Opacus
            try:
                network_wrap = GradSampleModule(network)
            except:
                network_wrap = network

            # Clear per-sample-grads
            del_grad_sample(network_wrap)

            # Out of safety, we also clear gradients
            network_wrap.zero_grad()

            # Get logits
            logits = network_wrap(inputs)
            if isinstance(logits, tuple):
                logits = logits[1]

            # Notice we should take the fact that averaging per-sample-gradient is needed into account
            (logits.sum()/inputs.size(0)).backward()

            ntk = 0

            # Here, we use weight (or also bias) to compute NTK, we exclude weight/bias of Norm layers
            for module in network.modules():
                if not isinstance(module, NORM_LAYERS):
                    if hasattr(module, 'weight') and hasattr(module.weight, 'grad'):
                        if 'weight' in params_types and (module.weight.grad is not None):
                            _grads = module.weight.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram

                    if hasattr(module, 'bias') and hasattr(module.bias, 'grad'):
                        if 'bias' in params_types and (module.bias.grad is not None):
                            _grads = module.bias.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram

            ntks.append(ntk)

            # Clear all grads
            del_grad_sample(network_wrap)
            network_wrap.zero_grad()

    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds

def get_analytical_metantk_n(xloader, networks, ways, algorithm ='ANIL', inner_lr_time = 1, reg_coef = 1e-3,
                            recalbn=0, train_mode=False, num_batch=-1, params_types='w', norm_type='groupnorm'):
    '''
    Here we compute analytical MetaNTK based on ANIL/MAML using Opacus.
    Notice: Since computational cost of NNGP is not that high, we will just compute NNGPs anyway
    :param xloader: A meta dataloader whose each batch contains support image, support labels, query images and query labels
    :param networks: A list of networks to compute MetaNTK condition numbers
    :param ways: Number of ways for few shot learning
    :param algorithm: algorithm used for computing MetaNTK, choices = ['ANIL', 'MAML']
    :param inner_lr_time: the product of inner loop learning rate & training time
    :param reg_coef: the regularization coefficient for the inner loop optimization. suggest >=1e-5
    :param recalbn: If recalculate batchnorm statistics
    :param train_mode: If enable batchnorm statistics to update during computing MetaNTK
    :param num_batch: Number of batches to average when computing MetaNTK
    :param params_types: parameters type for computing MetaNTK: 'w' for only weights, 'wb' for weights and bias
    :param norm_type: normalization layers used by the networks
    :return: the analytical MetaNTK condition numbers whose size is the same as size of networks
    '''
    device = torch.cuda.current_device()

    if params_types == 'w':
        params_types = ['weight',]
    elif params_types == 'wb':
        params_types = ['weight', 'bias']

    # If use inspector, one may find 'main' not supported
    # However, Opacus still works properly 
    # since we do not compute gradients of architecture parameters
    # inspector = DPModelInspector()

    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)

    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()

    ######

    ntks = []
    nngps = []

    for i, data in enumerate(xloader):

        # Only need images to compute NTKs
        support_xs, _, query_xs, _ = data
        if num_batch > 0 and i >= num_batch: break
        support_xs = support_xs.cuda(device=device, non_blocking=True)
        query_xs = query_xs.cuda(device=device, non_blocking=True)
        batch_size, n_support, channel, height, width = support_xs.size()
        _, n_queries, _, _, _ = query_xs.size()

        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        # print(support_xs.size())
        # print(query_xs.size())

        # run once to get dimension of output
        with torch.no_grad():

            # get first network to extract dimension of output
            network = networks[0]

            # Pass query images to get features
            query_features = network(query_xs)
            if isinstance(query_features, tuple):
                query_features = query_features[
                    0]  # 201 networks: return features and logits, here we use features

            temp_feature_size = query_features.size()
            assert len(
                temp_feature_size) == 2, "Model output should have dimension two which is (size of batch_size x output_dim)"
            
        xs = torch.cat([query_xs, support_xs], axis=0)

        for net_idx, network in enumerate(networks):

            network = network.to(device)

            # construct head
            head = torch.nn.Linear(temp_feature_size[1], ways)
            head.to(device)
            
            # Wrap network and head
            try:
                network_wrap = GradSampleModule(network)
            except:
                network_wrap = network
                
            head_wrap = GradSampleModule(head)

            # Clear per-sample-grads and grads
            network.zero_grad()
            head.zero_grad()
            del_grad_sample(network_wrap)
            del_grad_sample(head_wrap)

            features = network_wrap(xs)
            if isinstance(features, tuple):
                features = features[0]
            logit = head_wrap(features)

            # Notice we should take the fact that averaging per-sample-gradient is needed into account
            (logit.sum()/xs.size(0)).backward()

            ntk = 0
            nngp = 0

            # Here, we use weight (or also bias) to compute NTK, we exclude weight/bias of Norm layers
            for module in network.modules():
                if not isinstance(module, NORM_LAYERS):
                    if hasattr(module, 'weight') and hasattr(module.weight, 'grad'):
                        if 'weight' in params_types and (module.weight.grad is not None):
                            _grads = module.weight.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram

                    if hasattr(module, 'bias') and hasattr(module.bias, 'grad'):
                        if 'bias' in params_types and (module.bias.grad is not None):
                            _grads = module.bias.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram

            # Also take into account the params of head for computing MetaNTK
            for module in head.modules():
                if not isinstance(module, NORM_LAYERS):
                    if hasattr(module, 'weight') and hasattr(module.weight, 'grad'):
                        if 'weight' in params_types and (module.weight.grad is not None):
                            _grads = module.weight.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram
                            nngp += _gram

                    if hasattr(module, 'bias') and hasattr(module.bias, 'grad'):
                        if 'bias' in params_types and (module.bias.grad is not None):
                            _grads = module.bias.grad_sample
                            _grads = _grads.view(_grads.size(0),-1)
                            _gram = torch.einsum('nc,mc->nm', [_grads, _grads]).detach()
                            ntk += _gram
                            nngp += _gram

            # Only under the following condition NNGP will be equal to NTK:
            # The output of the network is zero
            if torch.allclose(nngp, ntk):
                with torch.no_grad():
                    testfeatures = network(xs.squeeze())
                    if isinstance(testfeatures, tuple):
                        testfeatures = testfeatures[0]
                print(ntk)
                assert torch.allclose(testfeatures, torch.zeros_like(testfeatures)), 'Please check bug, only when output is zero then NNGP=NTK'
                assert norm_type == 'nonorm'

            ntks.append(ntk.cpu().detach())
            nngps.append(nngp.cpu().detach())

            # print(ntk.size())

            network.zero_grad()
            head.zero_grad()
            torch.cuda.empty_cache()

    if algorithm == 'MAML':
        K = ntks.copy()
    else:
        K = nngps.copy()

    # Compute MetaNTK based on NTKs and NNGPS
    assert len(ntks) == len(K)
    conds = []
    for net_idx in range(len(ntks)):
        # We will pass if those ntk == nngp since the output is zero tensor
        if torch.allclose(ntks[net_idx], nngps[net_idx]):
            conds.append(100000.0)
        else:
            metantk_cond = compute_analytical_metantk_n(ntks[net_idx], K[net_idx], batch_size, n_queries, n_support, inner_lr_time, reg_coef, algorithm=algorithm)['eigenvalues']
            conds.append(np.nan_to_num((metantk_cond[-1]/ metantk_cond[0]), copy=True, nan=100000.0))
    return conds