# demo
import numpy as np
from skimage import io
import glob
from SOD_measures import compute_ave_MAE_of_methods
import os

## 0. =======set the data path=======
print("------0. set the data path------")

# >>>>>>> Follows have to be manually configured <<<<<<< #
data_name = 'TEST-HKUIS' # this will be drawn on the bottom center of the figures
data_dir = '../results/images/' # set the data directory,
                          # ground truth and results to-be-evaluated should be in this directory
                          # the figures of PR and F-measure curves will be saved in this directory as well
gt_dir = '/home/guest/PuYuan/datasets/dataset_test/ECSSD/gts' # set the ground truth folder name
# gt_dir = '/home/guest/PuYuan/datasets/dataset_test/DUTS-TE/gts' # set the ground truth folder name
# valset = ['ECSSD', 'HKU-IS', 'PASCALS', 'DUTOMRON', 'SOD', 'DUTS-TE']
# rs_dirs = ['run-0-DUTS-TE', 'run-4-DUTS-TE']
# rs_dirs = ['run-0-SOD', 'run-4-SOD']
# rs_dirs = ['run-0-DUTOMRON', 'run-4-DUTOMRON']
# rs_dirs = ['run-0-PASCALS', 'run-4-PASCALS']
# rs_dirs = ['run-0-HKU-IS', 'run-4-HKU-IS']
# rs_dirs = ['run-6-ECSSD','run-6-HKU-IS', 'run-6-PASCALS', 'run-6-DUTOMRON', 'run-6-SOD', 'run-6-DUTS-TE']
# rs_dirs = ['run-00135-DUTS-TE']
rs_dirs = ['run-0012-ECSSD']  # 'rs1' contains the result of method1
                        # 'rs2' contains the result of method 2
                        # we suggest to name the folder as the method names because they will be shown in the figures' legend
lineSylClr = ['b-','r-', 'g-'] # curve style, same size with rs_dirs
linewidth = [1,1,1] # line width, same size with rs_dirs
# >>>>>>> Above have to be manually configured <<<<<<< #

# gt_name_list = glob.glob(data_dir+gt_dir+'/'+'*.png') # get the ground truth file name list
name_list = sorted(os.listdir(gt_dir))
# print(name_list)   # ['0001.png', '0002.png', '0003.png', '0004.png', '0005.png']
gt_name_list = list(map(lambda x: os.path.join(gt_dir + '/' + x), name_list))
# print(gt_name_list)   # 带路径的label列表

## get directory list of predicted maps
rs_dir_lists = []
for i in range(len(rs_dirs)):
    rs_dir_lists.append(data_dir+rs_dirs[i]+'/')
# print('rs_dir_lists', rs_dir_lists)    # ['./test_data/rs1/', './test_data/rs2/']
print('\n')


## 1. =======compute the average MAE of methods=========
print("------1. Compute the average MAE of Methods------")
aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list, rs_dir_lists)
print('\n')
for i in range(0, len(rs_dirs)):
    print('>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.3f'%(rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i]))


## 2. =======compute the Precision, Recall and F-measure of methods=========
from SOD_measures import compute_PRE_REC_FM_of_methods,plot_save_pr_curves,plot_save_fm_curves

print('\n')
print("------2. Compute the Precision, Recall and F-measure of Methods------")
PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list,rs_dir_lists,beta=0.3)
for i in range(0,FM.shape[0]):
    print(">>", rs_dirs[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
print('\n')


## 3. =======Plot and save precision-recall curves=========
# print("------ 3. Plot and save precision-recall curves------")
# plot_save_pr_curves(PRE, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
#                     REC, # numpy array (num_rs_dir,255)
#                     method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
#                     lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
#                     linewidth = linewidth, # curve width, shape (num_rs_dir)
#                     xrange = (0.5,1.0), # the showing range of x-axis
#                     yrange = (0.5,1.0), # the showing range of y-axis
#                     dataset_name = data_name, # dataset name will be drawn on the bottom center position
#                     save_dir = data_dir, # figure save directory
#                     save_fmt = 'png') # format of the to-be-saved figure
# print('\n')
#
# ## 4. =======Plot and save F-measure curves=========
# print("------ 4. Plot and save F-measure curves------")
# plot_save_fm_curves(FM, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
#                     mybins = np.arange(0,256),
#                     method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
#                     lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
#                     linewidth = linewidth, # curve width, shape (num_rs_dir)
#                     xrange = (0.0,1.0), # the showing range of x-axis
#                     yrange = (0.0,1.0), # the showing range of y-axis
#                     dataset_name = data_name, # dataset name will be drawn on the bottom center position
#                     save_dir = data_dir, # figure save directory
#                     save_fmt = 'png') # format of the to-be-saved figure
# print('\n')

print('Done!!!')
