# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-15 11:54'

import sys, time
import numpy as np
from functions import *
from netlaprls1_0rev import *
from new_pairs import *
from trainmodel import TrainModel
from sklearn.preprocessing import MinMaxScaler


data_dir = os.path.abspath('..')
print data_dir

'''
    1. 将intMat中0的下标 1的下标 记录下来
    2. 使用模型预测0位置的得分
    3. 将预测矩阵中intMat为1的下标位置设置为0
    4. 归一化
    5. 取出得分并排序
    
'''

#
# def normalize(mat):
#     min = np.array(scores).min()
#     max = np.array(scores).max()
#     return [(m - min) / (max - min) for m in np.array(mat)]


trainmodel = TrainModel(dataset='e', seeds=[22, ], cvs=0)
trainmodel.train_md()
# index 0
drug_names, target_names = trainmodel.dt_name()
# x0, y0 为0的下标
x0, y0 = trainmodel.index_0()

# zip(x, y) 未标记的index 返回未标记位置预测的值
# scores = trainmodel.index_scores(x0, y0)

# 处理已知的过后
predict_mat = trainmodel.reset_index_1()
norm = MinMaxScaler()
predict_mat_norm = norm.fit_transform(predict_mat)
predict_mat_norm_flatten = np.array(predict_mat_norm).flatten()

scores = trainmodel.index_scores(predict_mat_norm_flatten, x0, y0)
# argsort函数 默认返回数索引值从小到大
# ii 表示将预测到的scores列表 进行降序

ii = np.argsort(scores)[::-1]
print len(ii)
# 取出所有未标记的index

# 预测返回的样本个数
predict_num = 1000
output_dir = os.path.join(os.path.abspath('../'), 'output')
print 'output_dir is ' + output_dir
# TODO 预测标靶对
# 预测标靶对
predict_pairs = [(drug_names[x0[i]], target_names[y0[i]], scores[i]) for i in ii[:predict_num]]
# np.set_printoptions(suppress=True)

for pair in predict_pairs:
    print pair

# new_dti_file = os.path.join(output_dir, "_".join(['netlaprls', dataset, "new_dti.txt"]))
# novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'biodb'))


