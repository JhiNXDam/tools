import torch
from nni.compression.pytorch.utils.counter import count_flops_params
from model.MyNet import MyNet
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info

model = MyNet()
stat(model, (3, 352, 352))

# input = torch.randn(1, 3, 352, 352)  # 输入数据
# flops, params = profile(model, inputs=(input,))
# print(f"FLOPs: {flops}")
# print(f"Parameters: {params}")
# input = torch.randn(1, 3, 352, 352)
# flops, params, results = count_flops_params(model, input)

# flops, params = get_model_complexity_info(model, (3, 352, 352), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
# print("Flops:", flops)
# print("Params: ", params)