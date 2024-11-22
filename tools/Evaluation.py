import numpy as np
import os
from measures import compute_ave_MAE_of_methods
from SOD_measures import compute_PRE_REC_FM_of_methods

# ==============================================路径设置==========================================================
dataset_path = 'E:/RSI_dataset/'  # 数据集保存路径
data_dir = 'E:/my_model/MSAAFNet/results/'  # 测试结果保存路径
gt_dir = 'EORSSD-gts'  # 标签名字
# 模型范围从 40 到 60
model_filenames = [f'MSAAFNet_EORSSD.pth.{i}' for i in range(40, 61)]  # 要评估的模型名字列表
# ==============================================路径设置==========================================================

# ==============================================评估结果==============================================================

rs_dirs = model_filenames  # 包含所有模型名字

lineSylClr = ['r-', 'b-', 'g-', 'y-', 'm-', 'c-', 'k-', 'b--', 'r--', 'g--', 'y--', 'm--', 'c--', 'k--']  # 不同模型曲线样式
linewidth = [2] * len(rs_dirs)  # 每个模型的线宽设置为2
# >>>>>>> 以上部分可以手动调整 <<<<<<< #

# 获取标签文件的名字列表
name_list = sorted(os.listdir(data_dir + gt_dir))
gt_name_list = list(map(lambda x: os.path.join(data_dir + gt_dir + '/' + x), name_list))

# 获取预测结果文件夹的路径列表
rs_dir_lists = [data_dir + rs_dir + '/' for rs_dir in rs_dirs]

## 1. =======计算每个方法的平均绝对误差 (MAE)=========
print("------1. 计算每个方法的平均 MAE------")
aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list, rs_dir_lists)

## 2. =======计算每个方法的精度、召回率和 F-Measure=========
print('\n')
print("------2. 计算每个方法的精度、召回率和 F-Measure------")
PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list, rs_dir_lists, beta=0.3)

# 打开一个新的 txt 文件，准备写入数据
with open(data_dir + 'results.txt', 'w') as file:
    # 写入每个模型的评估结果
    for i in range(len(rs_dirs)):
        # 构建每行的输出字符串
        result_str = '>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.4f, maxF->%.4f, meanF->%.4f\n' % (
            rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i], np.max(FM, 1)[i], np.mean(FM, 1)[i])
        # 写入到文件中
        file.write(result_str)
        file.write('\n')

# 通知用户结果已保存
print("结果已保存到results.txt文件中")
