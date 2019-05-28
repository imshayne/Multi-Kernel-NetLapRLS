# _*_coding:utf-8_*_
from netlaprls1_0rev import *

__author__ = 'mcy'
__date__ = '2019-04-29 10:29'

from trainmodel import TrainModel
import numpy as np
import matplotlib.pyplot as plt
import os
from  functions import *
# from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as PR
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, normalize
import warnings
warnings.filterwarnings('always')


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
datasets = ['e', 'gpcr', 'nr', 'ic']
accs = []
# for dataset in datasets:
#     trainmodel = TrainModel(dataset, seeds=[7771, 8367, 22, 1812, 4659])
#     trainmodel.train_md()
#     # 模型Accuracy概率 50 folds
#     acc = trainmodel.train_accuracy_md()
#     accs.append(acc)
#

# gpcr 4659 e 8367
trainmodel = TrainModel(dataset='e', seeds=[22,], cvs=1)
intMat, drugMat, targetMat = trainmodel.train_md()

# 调参 beta_t bata_d
# trainmodel.cv_eval()


# from matplotlib import rc
#
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


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
#
datasets = {'e', 'gpcr', 'nr', 'ic'}
BASE_DIR = os.path.abspath("../output")
read_dir = os.path.join(BASE_DIR, 'netlaprls_aupr_cvs1_' + 'gpcr' + '_default.txt')
# auc_vec = load_metric_vector(read_dir)


def plot_beta(datasets):
    fig  = plt.figure()
    ax = fig.add_subplot(1,1,1)

    beta_ts = np.logspace(-6, 3, 10, endpoint=True)
    beta_ds = np.logspace(-6, 3, 10)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    scores = []

    for dataset in datasets:
        read_dir = os.path.join(BASE_DIR, 'netlaprls_aupr_cvs1_' + dataset + '.txt')
        scores.append(load_metric_vector(read_dir))

    for beta_d, color in zip(beta_ds, colors):
        scores_x = []
        for sc in scores[0]:
            scores_x.append(scores[0])
        ax.plot(beta_d, scores[0][:1], label=r"$\beta_{t}=%s$" % beta_ts, color=color)

    ax.set_xlabel(r"$\beta_{d}$")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.set_title("NetLapRLS")
    plt.show()

# plot_beta(datasets)


# ROC
# t_scores_dir = os.path.join(BASE_DIR, "netlaprls_testdata__gpcr.txt")
# t_labels_dir = os.path.join(BASE_DIR, "netlaprls_testlabel__gpcr.txt")
# test_scores =
# test_labels =
# # print test_scores[0,:].shape
# print "*****"*2
# # Accuracy_Score = acc(test_labels[0, :], np.round(test_scores[0, :]), normalize=True)
# PR_score = PR(test_labels[0, :], np.round(test_scores[0, :]), average='weighted',
#               labels=np.unique(np.round(test_scores[0, :])))
# print PR_score
#
# print PR_score.shape
# # print Precision_Score
# print "ok"
#

# t_auc_dir = os.path.join(BASE_DIR, "netlaprls_auc_cvs1_gpcr_default.txt")
# t_aupr_dir = os.path.join(BASE_DIR, "netlaprls_aupr_cvs1_gpcr_default.txt")
# t_auc = load_metric_vector(t_auc_dir)
# t_aupr = load_metric_vector(t_aupr_dir)
# print t_auc.shape
# print t_aupr.shape
# print test_scores.shape


def Plot_PR_ROC(test_label, test_score):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1,2,1)
    precision = dict()
    recall = dict()
    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(test_label[i], test_score[i])
        if i==0:
            ax.plot(recall[i], precision[i], label="Fold %dst" % (i+1))
        elif i==0:
            ax.plot(recall[i], precision[i], label="Fold %dnd" % (i+1))
        elif i == 0:
            ax.plot(recall[i], precision[i], label="Fold %drd" % (i + 1))
        else:
            ax.plot(recall[i], precision[i], label="Fold %dth" % (i + 1))
    ax.set_xlabel("Recall Score")
    ax.set_ylabel("Precision Score")
    ax.set_title("P-R (Ion Channel)")
    legend = plt.legend(loc=3, title='10-Fold Cross Validation', shadow=True, fontsize='large', framealpha=1)
    legend.get_frame().set_facecolor('#f9f9f2')
    legend.get_title().set_fontsize(fontsize=20)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()
    # ax.text(0.9, 1.1, r'Ion Channel', fontdict={'size': 25, 'color': 'r'})


    ax = fig.add_subplot(1, 2, 2)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        min_max_scaler = MinMaxScaler()
        fpr[i], tpr[i], _ = roc_curve(test_labels[i], test_scores[i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
        if i == 0:
            ax.plot(fpr[i], tpr[i], label="Fold %dst, auc=%s" % (i + 1, roc_auc[i]))
        elif i == 1:
            ax.plot(fpr[i], tpr[i], label="Fold %dnd, auc=%s" % (i + 1, roc_auc[i]))
        elif i == 2:
            ax.plot(fpr[i], tpr[i], label="Fold %drd, auc=%s" % (i + 1, roc_auc[i]))
        else:
            ax.plot(fpr[i], tpr[i], label="Fold %dth, auc=%s" % (i + 1, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (Ion Channel)")
    legend = plt.legend(loc=4, title='10-Fold Cross Validation', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#f9f9f2')
    legend.get_title().set_fontsize(fontsize=20)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()


    # ax.text(10, 4, r'dddd', fontdict={'size': 100, 'color': 'r'})
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/ic.png", dpi=300, bbox_inches='tight')


# Plot_PR_ROC(test_labels, test_scores)


def plot_optimize(X, D, T, seeds):

    beta_ts = np.logspace(-6, 3, 10)
    beta_ds = np.logspace(-6, 3, 10)
    scores_aupr = []
    scores_auc = []
    for beta_t in beta_ts:
        for beta_d in beta_ds:
            model = NetLapRLS(beta_d=beta_d,beta_t=beta_t)
            # 10折交叉验证
            cv_data = cross_validation(X, seeds)

            aupr_vec, auc_vec, ts, tl = train(model, cv_data, X, D, T)
            # 每次求出平均值 用于plot
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            scores_aupr.append(aupr_avg)
            scores_auc.append(auc_avg)

    # plot
    beta_ts, beta_ds = np.meshgrid(beta_ts, beta_ds)
    scores_auprs = np.array(scores_aupr).reshape(beta_ts.shape)
    scores_aucs = np.array(scores_auc).reshape(beta_ts.shape)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(beta_ts,beta_ds,scores_auprs, rstride=1,cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\beta_{t}$")
    ax.set_ylabel(r"$\beta_{d}$")
    ax.set_zlabel(r"$score$")
    ax.set_title("Enzyme(AUPR)")
    plt.show()


plot_optimize(intMat, drugMat, targetMat, [7771, 8367, 22, 1812, 4659])
