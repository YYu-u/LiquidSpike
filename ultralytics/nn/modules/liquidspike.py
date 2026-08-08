# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import warnings
# from visualizer import get_local

from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint
from typing import List

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math
# __all__ = ('MS_GetT','MS_CancelT', 'MS_ConvBlock','MS_Block','MS_DownSampling',
#            'MS_StandardConv','SpikeSPPF','SpikeConv','MS_Concat','SpikeDetect'
#            ,'Ann_ConvBlock','Ann_DownSampling','Ann_StandardConv','Ann_SPPF','MS_C2f',
#            'Conv_1','BasicBlock_1','BasicBlock_2','Concat_res2','Sample','MS_FullConvBlock','MS_ConvBlock_resnet50','MS_AllConvBlock','MS_ConvBlock_res2net',)

decay = 0.25  # 0.25 # decay constants
#
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output

class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike2(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant2(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)

class MultiSpike1(nn.Module):

    class quant1(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class BNAndPadLayer(nn.Module):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                    self.bn.running_var + self.bn.eps))
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output


class Conv2d_bn(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))

class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepRepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.Conv2d(out_channel, out_channel, kernel_size, 1, 0, groups=out_channel, bias=False),
        )
        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = LTC_IPLIF_Node(c1=c1, max_spike=8, dampening=1.0)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x


class SpikeConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # 全局统一使用 LTC_IPLIF_Node，保证平滑的梯度回传
        self.lif = LTC_IPLIF_Node(c1=c1, init_tau=2.0, max_spike=8)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        # 严格遵守 Pre-activation: 模拟值输入 -> LIF发射脉冲 -> 卷积 -> BN输出模拟值
        x_spike = self.lif(x)
        #x_out = self.bn(self.conv(x_spike.flatten(0, 1))).reshape(T, B, -1, int(H / self.s), int(W / self.s))xiugai
        x_2d = self.bn(self.conv(x_spike.flatten(0, 1)))
        _, C_new, H_new, W_new = x_2d.shape  # 直接获取 2D 卷积后的真实高宽
        x_out = x_2d.reshape(T, B, C_new, H_new, W_new)
        return x_out


class SpikeConvWithoutBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = LTC_IPLIF_Node(c1=c1, max_spike=8, dampening=1.0)
        self.s = s

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)

        x_spike = self.lif(x)
        x_out = self.conv(x_spike.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x_out

class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act2_layer=nn.Identity, bias=False, kernel_size=3, padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding,
                                 groups=med_channels, bias=bias)
        self.pwconv3 = SepRepConv(med_channels, dim)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = LTC_IPLIF_Node(c1=dim, max_spike=8, dampening=1.0)
        self.lif2 = LTC_IPLIF_Node(c1=med_channels, max_spike=8, dampening=1.0)
        self.lif3 = LTC_IPLIF_Node(c1=med_channels, max_spike=8, dampening=1.0)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

class SepAllConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act2_layer=nn.Identity, bias=False, kernel_size=3, padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding,
                                 groups=med_channels, bias=bias)
        self.pwconv3 = SepRepConv(med_channels, dim)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = LTC_IPLIF_Node(c1=dim, max_spike=8, dampening=1.0)
        self.lif2 = LTC_IPLIF_Node(c1=med_channels, max_spike=8, dampening=1.0)
        self.lif3 = LTC_IPLIF_Node(c1=med_channels, max_spike=8, dampening=1.0)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):
        super().__init__()
        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio

        self.lif1 = LTC_IPLIF_Node(c1=input_dim, max_spike=8, dampening=1.0)

        hidden_dim = int(input_dim * mlp_ratio)
        self.lif2 = LTC_IPLIF_Node(c1=hidden_dim, max_spike=8, dampening=1.0)

        self.conv1 = RepConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = RepConv(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)

        # 引入残差缩放因子（初始化为很小的值），防止膜电位累加爆炸
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, C, H, W = x.shape

        # 稳压残差
        x = self.Conv(x) * self.gamma1 + x
        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)

        # 稳压残差
        x = x * self.gamma2 + x_feat
        return x

class MS_AllConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):
        super().__init__()
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        x = self.Conv(x) + x
        x_feat = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x
        return x

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = LTC_IPLIF_Node(c1=in_channels, max_spike=8, dampening=1.0)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x

class SpikeSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        # SpikeConv 已更新，直接调用
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))


class SpikeDetect(nn.Module):
    """YOLOv8 脉冲检测头实现"""
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 显式初始化实例属性，防止 apply 报错
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3),
                          SpikeConv(c2, c2, 3),
                          SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c3, 3),
                          SpikeConv(c3, c3, 3),
                          SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)

        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # ... [保留原有的 forward 和 bias_init 方法不变] ...

    # 🟢 确保这个函数存在且没有拼写错误
    def forward(self, x):
        """混合时序检测头 (Hybrid Temporal Readout)"""
        if x[0].dim() == 4:
            for i in range(len(x)):
                x[i] = x[i].unsqueeze(0)

        shape = x[0].mean(0).shape  # BCHW
        for i in range(self.nl):
            # 分别计算回归分支(cv2)和分类分支(cv3)
            box_seq = self.cv2[i](x[i])  # Shape: [T, B, 4*reg_max, H, W]
            cls_seq = self.cv3[i](x[i])  # Shape: [T, B, nc, H, W]

            # 🔥 核心创新：时空解耦读出机制 (Spatiotemporal Decoupled Readout)
            # 1. 边界框回归：依赖最终稳定状态，取最后时刻 (T=-1)
            box_out = box_seq[-1]
            # 2. 类别预测：依赖时间窗口内的脉冲证据累积，取均值
            cls_out = cls_seq.mean(0)

            # 拼接后进入 YOLO 原始逻辑
            x[i] = torch.cat((box_out, cls_out), 1)

        if self.training:
            return x

        # 验证或推理逻辑
        if self.dynamic or self.shape != shape:
            from ultralytics.utils.tal import make_anchors
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        from ultralytics.utils.tal import dist2bbox
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化 Detect() 的偏置"""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].conv.bias.data[:] = 1.0
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)




class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=1):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
        return x

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x

class MS_Concat(nn.Module):
    def __init__(self, dimension=2):
        super().__init__()
        self.d = dimension

    def forward(self, x):

        return torch.cat(x, self.d)


#新

class LTC_IPLIF_Node(nn.Module):
    def __init__(self, c1, init_tau=5.0, init_vth=1.0, max_spike=8, dampening=1.0):
        super().__init__()
        self.vth_param = nn.Parameter(torch.tensor(math.log(init_vth), dtype=torch.float))
        self.tau_gate = nn.Conv2d(c1, c1, kernel_size=1, groups=c1, bias=True)
        nn.init.constant_(self.tau_gate.bias, math.log((1.0 - 1.0 / init_tau) / (1.0 / init_tau)))
        self.spike_act = self.ArctanSpikeFunction(max_spike, alpha=2.0, scale=dampening)

    class ArctanSpikeFunction(nn.Module):
        def __init__(self, max_val, alpha, scale):
            super().__init__()
            self.max_val, self.alpha, self.scale = max_val, alpha, scale

        class _Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, max_val, alpha, scale):
                ctx.save_for_backward(input)
                ctx.alpha, ctx.scale, ctx.max_val = alpha, scale, max_val
                return torch.round(torch.clamp(input, min=0, max=max_val))

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                surrogate_grad = 1.0 / (1.0 + (ctx.alpha * (input - torch.round(input))).pow(2))
                return grad_output * surrogate_grad * ctx.scale * (
                        (input >= 0) & (input <= ctx.max_val)).float(), None, None, None

        def forward(self, x): return self._Func.apply(x, self.max_val, self.alpha, self.scale)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        x = x.to(self.tau_gate.weight.dtype)
        T, B, C, H, W = x.shape
        v_th = torch.clamp(self.vth_param.exp(), min=0.01).to(x.dtype)

        if T == 1:
            return self.spike_act(x[0] / v_th).unsqueeze(0)

        x_flat = x.flatten(0, 1)
        gate_base = self.tau_gate(x_flat).view(T, B, C, H, W)

        mem = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        spike = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        output = []
        for t in range(T):
            decay = torch.sigmoid(gate_base[t])

            mem = (mem - spike * v_th) * decay + x[t]
            spike = self.spike_act(mem / v_th)
            output.append(spike)
        return torch.stack(output)


class LiquidSSM_Block(nn.Module):
    def __init__(self,channels,dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.local_mixer = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim, bias=False)
        self.norm_mixer = nn.BatchNorm3d(dim)

        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.alpha_fwd = nn.Parameter(torch.tensor([0.5]))
        self.alpha_bwd = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)
        self.pre_spike_norm = nn.BatchNorm2d(dim)

        self.out_lif= LTC_IPLIF_Node(c1=channels, init_tau=2.0, max_spike=8)


    def forward(self, x):
        if x.dim() == 4: return x
        x = x.to(self.local_mixer.weight.dtype)
        T, B, C, H, W = x.shape

        # === 1. 局部时空混合 ===
        x_3d = x.permute(1, 2, 0, 3, 4).contiguous()
        if torch.isnan(x_3d).any() or torch.isinf(x_3d).any():
            x_3d = torch.nan_to_num(x_3d, nan=0.0, posinf=1.0, neginf=-1.0)

        #x_mixed = self.norm_mixer(self.local_mixer(x_3d))修改
        x_3d = x_3d.contiguous()
        x_mixed = self.norm_mixer(self.local_mixer(x_3d))
        del x_3d

        x_mixed = x_mixed.permute(2, 0, 1, 3, 4).contiguous()
        x_flat = x_mixed.flatten(0, 1)
        x_seq = x_flat.flatten(2).transpose(1, 2)
        del x_mixed, x_flat

        # === 2. 双向 Mamba 扫描 ===
        out_fwd = self.mamba(x_seq)
        x_seq_bwd = torch.flip(x_seq, dims=[1])
        out_bwd = self.mamba(x_seq_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])
        del x_seq, x_seq_bwd

        x_out = out_fwd * self.alpha_fwd + out_bwd * self.alpha_bwd
        del out_fwd, out_bwd

        # === 3. 维度还原与稳压 ===
        x_out = x_out.to(x.dtype)
        x_out = x_out.transpose(1, 2).reshape(T * B, C, H, W)
        x_out = self.pre_spike_norm(x_out)

        # === 4. 残差连接 ===
        x_orig = x.flatten(0, 1)
        res = x_out * self.gamma.to(x.dtype).view(1, -1, 1, 1) + x_orig

        # 直接输出模拟值，交给下游处理
        return res.view(T, B, C, H, W)


class SpikingDynamicRouting(nn.Module):
    def __init__(self,channels, dim):
        super().__init__()
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(dim // 4, 2, 1, bias=True),
            nn.Softmax(dim=1)
        )
        self.alpha_param = nn.Parameter(torch.zeros(1))
        self.out_lif= LTC_IPLIF_Node(c1=channels, init_tau=2.0, max_spike=8)

    def forward(self, x):
        if isinstance(x, torch.Tensor): return x
        x_shallow, x_deep = x
        T, B, C, H, W = x_shallow.shape

        alpha = torch.sigmoid(self.alpha_param)
        fused = (x_shallow * alpha + x_deep * (1 - alpha)).flatten(0, 1).contiguous()

        weights = self.router(fused).view(T, B, 2, 1, 1, 1)
        out = weights[:, :, 0] * x_shallow + weights[:, :, 1] * x_deep


        return out


class SpikingEMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(SpikingEMA, self).__init__()
        self.groups = max(1, channels // factor)

        #在模块最前端添加统一的脉冲转换节点
        self.in_lif = LTC_IPLIF_Node(c1=channels, init_tau=2.0, max_spike=8)

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.aggr = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(0)
        x = x.to(self.conv1x1.weight.dtype)
        T, B, C, H, W = x.shape

        # 1. 模拟值转换为脉冲
        spike_in = self.in_lif(x).flatten(0, 1)

        # 2. 提取特征 (这里输出的是模拟值)
        feat1 = self.bn1(self.conv1x1(spike_in))
        feat2 = self.bn2(self.conv3x3(spike_in))

        # 3. 计算注意力
        x_h = self.pool_h(feat1)
        x_w = self.pool_w(feat2).permute(0, 1, 3, 2)
        attn = self.aggr(torch.cat([x_h, x_w], dim=2))

        a_h, a_w = torch.split(attn, [H, W], dim=2)
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w.permute(0, 1, 3, 2))

        # 4. 加权特征并加入残差缩放，输出模拟值
        x_flat = x.flatten(0, 1)
        out = (feat1 * a_h * a_w).reshape(T, B, C, H, W)

        return out * self.gamma + x


class SpikeDFL(nn.Module):

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class LiquidSSM_Block(nn.Module):
    def __init__(self,channels, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.local_mixer = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim, bias=False)
        self.norm_mixer = nn.BatchNorm3d(dim)

        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.alpha_fwd = nn.Parameter(torch.tensor([0.5]))
        self.alpha_bwd = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)
        self.pre_spike_norm = nn.BatchNorm2d(dim)

        self.out_lif= LTC_IPLIF_Node(c1=channels, init_tau=2.0, max_spike=8)

    def forward(self, x):
        # 0. 拦截非时序输入
        if x.dim() != 5:
            return x

        x = x.to(self.local_mixer.weight.dtype)
        T, B, C, H, W = x.shape

        # === 1. 局部时空混合 (Local Spatiotemporal Mixing) ===
        # 🚨 关键修复1：必须使用 .contiguous() 强制内存连续！
        # 维度转换: [T, B, C, H, W] -> [B, C, T, H, W] (符合 Conv3d 标准要求)
        x_3d = x.permute(1, 2, 0, 3, 4).contiguous()

        # 安全钳制（防止边界异常值）
        if torch.isnan(x_3d).any() or torch.isinf(x_3d).any():
            x_3d = torch.nan_to_num(x_3d, nan=0.0, posinf=1.0, neginf=-1.0)

        # 执行 3D 卷积和归一化
        x_mixed = self.norm_mixer(self.local_mixer(x_3d))


        x_mixed = x_mixed.permute(2, 0, 1, 3, 4).contiguous()

        x_flat = x_mixed.view(T, B, C, H * W)

        # 维度转换: [T, B, C, H*W] -> [T*B, C, H*W] -> [T*B, H*W, C] (Mamba 要求的格式)
        x_seq = x_flat.flatten(0, 1).transpose(1, 2)

        # === 3. 双向 Mamba 扫描 ===
        out_fwd = self.mamba(x_seq)

        x_seq_bwd = torch.flip(x_seq, dims=[1])
        out_bwd = self.mamba(x_seq_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        x_out = out_fwd * self.alpha_fwd + out_bwd * self.alpha_bwd

        # === 4. 还原维度与稳压残差 ===
        x_out = x_out.to(x.dtype)

        # 维度还原: [T*B, H*W, C] -> [T*B, C, H*W] -> [T*B, C, H, W]
        x_out = x_out.transpose(1, 2).view(T * B, C, H, W)

        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            x_out = torch.nan_to_num(x_out, nan=0.0, posinf=1.0, neginf=-1.0)

        x_out = self.pre_spike_norm(x_out)

        # 还原时间步维度 T
        x_out = x_out.view(T, B, C, H, W)

        safe_gamma = torch.nan_to_num(self.gamma, nan=0.0)

        # 修正 gamma 的广播维度并进行残差相加
        res = x_out * safe_gamma.to(x.dtype).view(1, 1, -1, 1, 1) + x
        if torch.isnan(res).any() or torch.isinf(res).any():
            res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)

        res = torch.clamp(res, min=-10.0, max=10.0)

        return res


class LiquidSSM_Block(nn.Module):
    def __init__(self,channels, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.local_mixer = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim, bias=False)
        self.norm_mixer = nn.BatchNorm3d(dim)

        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.alpha_fwd = nn.Parameter(torch.tensor([0.5]))
        self.alpha_bwd = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)
        self.pre_spike_norm = nn.BatchNorm2d(dim)
        self.out_lif= LTC_IPLIF_Node(c1=channels, init_tau=2.0, max_spike=8)

    def forward(self, x):
        # 0. 拦截非时序输入
        if x.dim() != 5:
            return x

        x = x.to(self.local_mixer.weight.dtype)
        T, B, C, H, W = x.shape

        # === 1. 局部时空混合 (Local Spatiotemporal Mixing) ===

        x_3d = x.permute(1, 2, 0, 3, 4).contiguous()

        # 🌟 ========================================== 🌟
        # 🌟 新增：动态 Padding 护盾 (专门对付验证集中的细长变态图片) 🌟
        # 获取 x_3d 在当前状态下的空间维度 (高度和宽度)
        curr_h, curr_w = x_3d.shape[3], x_3d.shape[4]


        pad_h = max(0, 3 - curr_h)
        pad_w = max(0, 3 - curr_w)

        if pad_h > 0 or pad_w > 0:

            x_3d = F.pad(x_3d, (0, pad_w, 0, pad_h), mode='constant', value=0.0)


        # 安全钳制（防止边界异常值）
        if torch.isnan(x_3d).any() or torch.isinf(x_3d).any():
            x_3d = torch.nan_to_num(x_3d, nan=0.0, posinf=1.0, neginf=-1.0)

        # 执行 3D 卷积和归一化
        x_mixed = self.norm_mixer(self.local_mixer(x_3d))
        if pad_h > 0 or pad_w > 0:
            x_mixed = x_mixed[:, :, :, :curr_h, :curr_w].contiguous()
        x_mixed = x_mixed.permute(2, 0, 1, 3, 4).contiguous()
        x_flat = x_mixed.view(T, B, C, H * W)
        x_seq = x_flat.flatten(0, 1).transpose(1, 2)

        # === 3. 双向 Mamba 扫描 ===
        out_fwd = self.mamba(x_seq)

        x_seq_bwd = torch.flip(x_seq, dims=[1])
        out_bwd = self.mamba(x_seq_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        x_out = out_fwd * self.alpha_fwd + out_bwd * self.alpha_bwd

        # === 4. 还原维度与稳压残差 ===
        x_out = x_out.to(x.dtype)

        # 维度还原: [T*B, H*W, C] -> [T*B, C, H*W] -> [T*B, C, H, W]
        x_out = x_out.transpose(1, 2).view(T * B, C, H, W)

        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            x_out = torch.nan_to_num(x_out, nan=0.0, posinf=1.0, neginf=-1.0)

        x_out = self.pre_spike_norm(x_out)

        # 还原时间步维度 T
        x_out = x_out.view(T, B, C, H, W)

        safe_gamma = torch.nan_to_num(self.gamma, nan=0.0)

        res = x_out * safe_gamma.to(x.dtype).view(1, 1, -1, 1, 1) + x

        if torch.isnan(res).any() or torch.isinf(res).any():
            res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)

        res = torch.clamp(res, min=-10.0, max=10.0)

        return res