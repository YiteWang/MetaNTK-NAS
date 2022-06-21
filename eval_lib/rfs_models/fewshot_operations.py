import torch
import torch.nn as nn

OPS = {
    'none'        : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats),
    'max_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'max', affine, track_running_stats),
    'avg_pool_3x3BN': lambda C_in, C_out, stride, affine, track_running_stats: PoolBN("avg", C_in, 3, stride, 1, affine, track_running_stats),
    'max_pool_3x3BN': lambda C_in, C_out, stride, affine, track_running_stats: PoolBN("max", C_in, 3, stride, 1, affine, track_running_stats),
    'nor_conv_7x7': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1), affine, track_running_stats),
    'nor_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (5,5), (stride,stride), (2,2), (1,1), affine, track_running_stats),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats:     SepConv(C_in, C_out, (3,3), (stride,stride), (2,2), (2,2), affine, track_running_stats),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine, track_running_stats:     SepConv(C_in, C_out, (5,5), (stride,stride), (4,4), (2,2), affine, track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
    'conv_7x1_1x7': lambda C_in, C_out, stride, affine, track_running_stats: FacConv(C_in, C_out, 7, stride, 3, affine=affine),
    'conv_1x5_5x1': lambda C_in, C_out, stride, affine, track_running_stats: FacConv(C_in, C_out, 5, stride, 2, affine=affine),
    'conv_3x3'    : lambda C_in, C_out, stride, affine, track_running_stats:     SepConv(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
}

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x

class StdConv(nn.Module):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.net(x)

class DropPath_(nn.Module):
    def __init__(self, p=0.0):
        """[!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return f"p={self.p}, inplace"

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(C_in, C_in , kernel_size, stride, padding, dilation, affine, track_running_stats)
        self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats)

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                                                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                                      nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim  = inplanes
        self.out_dim = planes
        self.stride  = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess: x = self.preprocess(inputs)
        else              : x = inputs
        return self.op(x)

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN,
    get from MetaNAS
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True, track_running_stats=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False
            )
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats)
        # self.bn = BatchNorm_scratch(C, 4)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in   = C_in
        self.C_out  = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1: return x.mul(0.)
            else               : return x[:,:,::self.stride,::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in   = C_in
        self.C_out  = C_out
        self.relu   = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append( nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False) )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)

class FacConv(nn.Module):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True, track_running_stats=True):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                C_in, C_out, (1, kernel_length), stride, (0, padding), bias=False
            ),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), 1, (padding, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.net(x)

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)