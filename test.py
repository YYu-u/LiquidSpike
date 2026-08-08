import os
from ultralytics import YOLO
import torch.nn as nn
import torch
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from ultralytics.nn.modules.yolo_spikformer import LTC_IPLIF_Node  # 导入你的自定义类

fr_dict = {}
iter = 125  # the iters of an epoch

i = 0

model = YOLO('/path/to/checkpoint/.pt')


def forward_hook_fn(module, input, output):  # 计算每一层的发放率
    global i
    if module.name == 'model.model.2.Conv.lif1':
        i = i + 1
        print("i:", i)
        # print(output.shape)
    x_shape = torch.tensor(list(output.shape))

    if module.name not in fr_dict.keys():

        fr_dict[module.name] = output.detach().mean() / iter
    else:

        fr_dict[module.name] = fr_dict[module.name] + output.detach().mean() / iter


for n, m in model.named_modules():
    if isinstance(m, (MultiStepLIFNode, LTC_IPLIF_Node)):
        print(n)
        m.name = n
        m.register_forward_hook(forward_hook_fn)

model.val(data="gen1.yaml", device=[0,1,2,3])
print("fire:", fr_dict)  # the firing rate of each layer

# 测试模型

