# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-15 12:10'

import time
from functions import *
from netlaprls1_0rev import NetLapRLS
import pickle

# TODO xyz  y-- 10-5 10-3 10
# TODO beta_d beta_t 生成矩阵 可视化图
# TODO


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.logspace(-6, 3, 10):  # [-6, 2]
        for y in np.logspace(-6, 3, 10):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10, gamma_t=10, beta_d=x, beta_t=y)
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print cmd
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print "auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])