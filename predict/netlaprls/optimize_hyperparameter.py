# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-24 18:56'

import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as np
import os
from functions import *
from netlaprls import *

from trainmodel import *
datasets = {'e', 'gpcr', 'nr', 'ic'}
BASE_DIR = os.path.abspath("../output")
read_dir = os.path.join(BASE_DIR, 'netlaprls_aupr_cvs1_eall_auc_aupr_optimize.txt')
aupr_all_100 = load_metric_vector(read_dir)
trainmodel = TrainModel(dataset='e', seeds=[1812, ], cvs=1)
intMat, drugMat, targetMat = trainmodel.train_md()


def train_optimize(beta_d, beta_t, inMat=intMat, drugMat=drugMat, targetMat=targetMat, seeds=[1812, ], cv=1):
    model = NetLapRLS(beta_d=beta_d, beta_t=beta_t)
    cv_data = cross_validation(intMat, seeds, cv=1)
    aupr_vec, auc_vec, test_scores, test_label = train(model, cv_data, intMat, drugMat, targetMat)

    return aupr_vec


def Plot_opt_para(aupr_all):
    # beta_d axis 0 beta_t axis 1
    beta_ds = np.logspace(-6, 3, 10)
    beta_ts = np.logspace(-6, 3, 10)

    beta_ds, beta_ts = np.meshgrid(beta_ds, beta_ts)
    # optimal = train_optimize(beta_ds, beta_ts)
    optimal = aupr_all
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(beta_ds, beta_ts, marker='.', color='blue', linestyle='none')
    ax.contour(beta_ds, beta_ts, optimal, 20,  cmp='RdGy')
    ax.colorbar()
    # ax.set_ylim(beta_ts)
    # plt.yticks(beta_ts)
    # ax.set_yticklabels(beta_ts)
    # # ax.set_xlim(beta_ds)
    # plt.xticks(beta_ds)
    # ax.set_xticklabels(beta_ds)


    plt.show()
    # ax.set_yticks(range(len(beta_t)))
    # ax.set_yticklabels(beta_t)
    # ax.set_xticks(range(len(beta_d)))
    # ax.set_xticklabels(beta_d)

    # im = ax.imshow(, cmap=plt.cm.hot_r)

Plot_opt_para(aupr_all_100)