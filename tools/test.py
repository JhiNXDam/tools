import torch
import torch.nn.functional as F

import numpy as np
import os
import argparse
import imageio
from model.MSAAFNet import MSAAFNet
from utils1.data_edge import test_dataset
from measures import compute_ave_MAE_of_methods
from SOD_measures import compute_PRE_REC_FM_of_methods

# Set the CUDA device
torch.cuda.set_device(0)

# Argument parser for test size
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
args = parser.parse_args()
opt = args

# ==============================================路径设置==========================================================
dataset_path = 'E:/RSI_dataset/'  # 数据集保存路径
data_dir = 'E:/my_model/MSAAFNet/results/'  # 测试结果保存路径
gt_dir = 'EORSSD-gts'  # 标签名字
models_path = 'E:/my_model/MSAAFNet/models/'  # 预训练模型保存路径
filenames = [f'MSAAFNet_EORSSD.pth.{i}' for i in range(52, 53)]  # 预训练模型名字
test_datasets = ['EORSSD']  # 测试数据集名字
# ==============================================路径设置==========================================================

# ==============================================开始测试==========================================================
model = MSAAFNet()
for filename in filenames:
    model_filename = models_path + filename
    model.load_state_dict(torch.load(model_filename))
    model.cuda()
    model.eval()

    for dataset in test_datasets:
        save_path = data_dir + filename + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/test-images/'
        print(f'Testing {dataset} with model {model_filename}')
        gt_root = dataset_path + dataset + '/test-labels/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4, s5, s5_sig, edg5 = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imsave(save_path + name, res)

#==============================================评估结果==============================================================

rs_dirs = filenames  # Include both filenames

lineSylClr = ['r-', 'b-']  # curve style, same size with rs_dirs
linewidth = [2, 1]  # line width, same size with rs_dirs
# >>>>>>> Above have to be manually configured <<<<<<< #

# Get the ground truth file name list
name_list = sorted(os.listdir(data_dir + gt_dir))
gt_name_list = list(map(lambda x: os.path.join(data_dir + gt_dir + '/' + x), name_list))

# Get directory list of predicted maps
rs_dir_lists = [data_dir + rs_dir + '/' for rs_dir in rs_dirs]

## 1. =======compute the average MAE of methods=========
print("------1. Compute the average MAE of Methods------")
aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list, rs_dir_lists)

## 2. =======compute the Precision, Recall and F-measure of methods=========
print('\n')
print("------2. Compute the Precision, Recall and F-measure of Methods------")
PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list, rs_dir_lists, beta=0.3)

# Open a new txt file, prepare to write data
with open(data_dir + 'results.txt', 'w') as file:
    # Write the computed results for each model
    for i in range(len(rs_dirs)):
        # Construct each line of output string
        result_str = '>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.4f, maxF->%.4f, meanF->%.4f\n' % (
            rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i], np.max(FM, 1)[i], np.mean(FM, 1)[i])
        # Write to file
        file.write(result_str)
        file.write('\n')

# Notify the user that the results have been saved
print("结果已保存到results.txt文件中")
