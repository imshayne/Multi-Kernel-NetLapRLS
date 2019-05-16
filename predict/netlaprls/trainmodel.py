# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-15 10:09'
import pickle
from functions import *
from netlaprls1_0rev import *
import  numpy as np
import time
import os
from cv_eval import *

# 读路径
data_dir = os.path.abspath('..')
print data_dir

# 取出数据集
dataset = 'e'

# inMat作用矩阵 drugMat药物相似矩阵 targetMat标靶相似矩阵
intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
# 返回drug target的名称
drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))

model = NetLapRLS()
seeds = [22, ]
cv_data = cross_validation(intMat, seeds, cv=1)
tic = time.clock()
aupr_vec, auc_vec = train(model, cv_data, intMat, drugMat, targetMat)

aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)

auc_avg, auc_conf = mean_confidence_interval(auc_vec)

print "auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" \
      % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)

output_dir = os.path.join(os.path.abspath('../'), 'output')
print 'output_dir is ' + output_dir
# 生成auc apcr 参数
# TODO auc apcr 参数生成
# method = "netlaprls"
# cvs = 1
# write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method + "_auc_cvs" + str(cvs) + "_" + dataset + ".txt"))
# print 'Ok'
# write_metric_vector_to_file(aupr_vec,os.path.join(output_dir, method + "_aupr_cvs" + str(cvs) + "_" + dataset + ".txt"))

# 模型调参
cvs = 1
# args 模型参数ValueError: Integers to negative integer powers are not allowed.ValueError: Integers to negative integer powers are not allowed.

# TODO args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
args = None

netlaprls_cv_eval('netlaprls', dataset, cv_data, intMat, drugMat, targetMat, cvs, args)


