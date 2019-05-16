# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-15 11:54'

import sys, time
import numpy as np
from functions import *
from netlaprls1_0rev import *
from new_pairs import *

data_dir = os.path.abspath('..')
print data_dir
# 取出数据集
dataset = 'e'

# inMat作用矩阵 drugMat药物相似矩阵 targetMat标靶相似矩阵
intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
# 返回drug target的名称
drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))

dataset = 'e'
model = NetLapRLS()
print "Dataset:" + dataset + "\n" + 'netlaprls'
seed = [22, ]

model.fix_model(intMat, intMat, drugMat, targetMat, seed)
x, y = np.where(intMat == 0)
scores = model.predict_scores(zip(x, y), 5)
# argsort函数 返回数组值从大到小的索引值
ii = np.argsort(scores)[::-1]
# 预测返回的样本个数
predict_num = None

output_dir = os.path.join(os.path.abspath('../'), 'output')
print 'output_dir is ' + output_dir
# TODO 预测标靶对
# 预测标靶对
predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii[:predict_num]]
new_dti_file = os.path.join(output_dir, "_".join(['netlaprls', dataset, "new_dti.txt"]))
novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'biodb'))
