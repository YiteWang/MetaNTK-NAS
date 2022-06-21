""" CNN for network augmentation 
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""


""" 
Based on https://github.com/khanrc/pt.darts
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""

import torch
import torch.nn as nn

from .fewshot_operations import *

def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = OPS[op_name](C_in, C_in, stride, True, True)
            if not isinstance(op, Identity):  # Identity does not use drop path
                op = nn.Sequential(op, DropPath_())
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag

class AugmentCNN(nn.Module):
    """ Augmented CNN model """

    def __init__(
        self,
        input_size,
        C_in,
        C,
        n_classes,
        n_layers,
        auxiliary,
        genotype,
        stem_multiplier=3,
        feature_scale_rate=2,
        reduction_layers=[],
    ):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False), nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False

        if not reduction_layers:
            reduction_layers = [n_layers // 3, (2 * n_layers) // 3]

        for i in range(n_layers):
            if i in reduction_layers:
                C_cur *= feature_scale_rate
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size // 4, C_p, n_classes)

        # self.lastact    = nn.Sequential(nn.BatchNorm2d(C_p), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, n_classes)

        self.criterion = nn.CrossEntropyLoss()

        ####### dummy alphas
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(2):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(1, 5)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(1, 5)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.alpha_prune_threshold = 0.0

    def forward(self, x, is_feat=False):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        # out = self.lastact(s1)
        # out = self.gap(out)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.classifier(out)

        if is_feat:
            return [out], logits
        else:
            if self.aux_pos == -1:  # no auxiliary head
                return logits
            else:
                return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

    def weights(self):
        return self.parameters()

    def named_weights(self):
        return self.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p
        # return None

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
        # return None

    def genotype(self):
        return None

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def get_sparse_num_params(
        self, alpha_prune_threshold=0.0
    ):  # dummy function to not break code
        """Get number of parameters for sparse one-shot-model (in this case just number of parameters of model)

        Returns:
            A torch tensor
        """
        return None


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            # 2x2 out
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 1x1 out
            nn.Conv2d(128, 768, kernel_size=2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.classifier(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.classifier(out)
        return logits


class AugmentCell(nn.Module):
    """Cell for augmentation
    Each edge is discrete.
    """

    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)

        if reduction_p:
            self.preproc0 = FactorizedReduce(C_in=C_pp, C_out=C, stride=2, affine=True, track_running_stats=True)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out
