# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 10:29'

from trainmodel import TrainModel
import numpy as np
import matplotlib.pyplot as plt

############################################################################
# # defaultdict 实例
# from collections import defaultdict
# s = [('Python', 1), ('Swift', 2), ('Python', 3), ('Swift', 4), ('Python', 9)]
# # 创建defaultdict，设置由list()函数来生成默认值
# d = defaultdict(list)
# for k, v in s:
#     # 直接访问defaultdict中指定key对应的value即可。
#     # 如果该key不存在，defaultdict会自动为该key生成默认值
#     d[k].append(v)
# print(list(d.items()))
# print len(d)

############################################################################
# 模型运行 未进行调参 默认值训练
# datasets = ['e', 'gpcr', 'nr', 'ic']
# accs = []
# for dataset in datasets:
#     trainmodel = TrainModel(dataset, seeds=[7771, 8367, 22, 1812, 4659])
#     trainmodel.train_md()
#     # 模型Accuracy概率 50 folds
#     acc = trainmodel.train_accuracy_md()
#     accs.append(acc)
#
trainmodel = TrainModel(dataset='e', seeds=[7771, 8367, 22, 1812, 4659], cvs=1)
trainmodel.train_md()

# 调参 beta_t bata_d
trainmodel.cv_eval()


from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def make_plot(axs, accs):
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    # Fixing random state for reproducibility

    ax1 = axs[0, 0]
    ax1.plot(accs[0])
    ax1.set_title('accuracy')
    ax1.set_ylabel('e', bbox=box)
    ax1.set_ylim(0, 1)

    ax3 = axs[1, 0]
    ax3.set_ylabel('gpcr', bbox=box)
    ax3.plot(accs[1])
    ax3.set_ylim(0, 1)

    ax2 = axs[0, 1]
    ax2.set_title('10 fold x5')
    ax2.plot(accs[2])
    ax2.set_ylabel('nr', bbox=box)
    ax2.set_ylim(0, 1)

    ax4 = axs[1, 1]
    ax4.plot(accs[3])
    ax4.set_ylabel('ic', bbox=box)
    ax4.set_ylim(0, 1)


# Plot 1:
# fig, axs = plt.subplots(2, 2)
# fig.subplots_adjust(left=0.2, wspace=0.6)
# make_plot(axs,accs)
#
# # just align the last column of axes:
# fig.align_ylabels(axs[:, 1])
# plt.show()

######################################################
# Create a legend and tweak it with a shadow and a box.
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Cross Validation Curve With NetLapRLS')
# ax.set_ylabel(u'Accuracy  Scores')
# ax.set_ylim(0.5, 1)
# ax.set_xlabel(u'$K$ Fold')
# ax.set_xlim(1, 50)
# t1 = np.arange(0.0, 1.0, 0.01)
# for i, n in enumerate(accs):
#     plt.plot(np.arange(1, 51), n, label="%s" % (datasets[i],))
#
# leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
# leg.get_frame().set_alpha(0.5)
#
# plt.show()
# fig.savefig("../output/AccuracyScore.png", dpi=300, bbox_inches='tight')
