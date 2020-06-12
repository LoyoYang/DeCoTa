import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import os


# def plot_confusion_matrix(df_confusion):
#     side_l = df_confusion.shape[0]
#     df_cm = pd.DataFrame(df_confusion, range(side_l), range(side_l))
#     sn.set(font_scale=0.1)
#     sn.heatmap(df_cm, annot=False)
#     plt.savefig('../transfer/{}_rank.png'.format(file_name))


# file_name = 'real_to_clipart_step500_runs_56_test'
# test_mat = np.load('./tmp/cfs/{}.npy'.format(file_name))
#
# # di = np.diag_indices(test_mat.shape[0])
# # di_value = test_mat[di]
# # di_rank = np.flip(np.argsort(di_value))
# # test_mat = test_mat[di_rank, :]
# plot_confusion_matrix(test_mat)

file_dict = [56, 57, 58, 59, 60, 61]
exp_names = ['s+t', 's+t+ent', 's+t+mix(t,u)', 's+t+mix(s,u)', 'correct_pseudo', 'wrong_pseudo']
colors = ['red', 'orange', 'blue', 'black', 'green', 'yellow']
steps = np.arange(30) * 1000
cfs_ratio = 0.5

fig, ax = plt.subplots()
t = np.arange(0, steps.shape[0], 1)

for i, exp in enumerate(file_dict):
    line = np.zeros(steps.shape[0])
    for idx, step in enumerate(steps):
        file_root = '/private/home/yangluyu/source/yly/code/DA/SSDA_MME_mod/tmp/cfs'
        cfs_mat_file = os.path.join(file_root,
                                    'real_to_clipart_step{}_runs_{}_test.npy'.format(step, exp))
        test_mat = np.load(cfs_mat_file)
        test_mat = np.sort(test_mat, axis=1)
        out = np.sum(test_mat[:,:-2], axis=1)/test_mat[:,-1] >= 0.5
        line[idx] = out.sum()
    ax.plot(steps, line, color=colors[i], linewidth=1.5, label='{}'.format(exp_names[i]))
    ax.legend(loc='best')
    ax.set(xlabel='iter', ylabel='num_cls above ratio')

save_file = '/private/home/yangluyu/source/yly/code/DA/transfer/y_cfs_classes_x_iter.png'
fig.savefig(save_file)