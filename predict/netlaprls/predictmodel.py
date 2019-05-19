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


# 设置保存的路径
data_dir = os.path.abspath('..')
print data_dir
output_dir = os.path.join(os.path.abspath('../output'), 'new_dti')
print 'output_dir is ' + output_dir

'''
    1. 将intMat中0的下标 1的下标 记录下来
    2. 使用模型预测0位置的得分
    3. 将预测矩阵中intMat为1的下标位置设置为0
    4. 归一化
    5. 取出得分并排序
    
    prime parameters:
        'e'
        'gpcr'
        'ic'
        'nr' 
'''


# 将intMat中1位置相对于predictR位置上的预测值 设置为较小值
# 确保不影响 预测值
def set_1_litte(intMat, pR, value=1e-5):
    x1, y1 = np.where(intMat == 1)
    pR_ = pR
    for x, y in zip(x1, y1):
        pR_[x, y] = value
    return pR_


# 归一化矩阵 显示0-1
def norm(mat):
    norm = MinMaxScaler()
    return norm.fit_transform(mat)


# 返回指定预测值
def re_scores(mat, x, y):
    inx = np.array(zip(x,y))
    return mat[inx[:, 0], inx[:, 1]]


# 选择数据集
'''
    datasets = {'e', 'gpcr', 'nr', 'ic'}
    
    'e':
    Best beta_d=0.000001, beta_t=0.000001.
    prime aupr 0.785952

    
'gpcr':
    Best beta_d=0.000010, beta_t=0.000010.
    prime aupr 0.617384

'nr':
    Best beta_d=0.100000, beta_t=0.010000.
    prime aupr 0.462829

'ic':
    Best beta_d=0.000001, beta_t=0.000001.
    prime aupr 0.820125

'''
dataset = 'gpcr'

intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))
# 使用默认参数fix
model = NetLapRLS(beta_d=0.000010, beta_t=0.000010)

# 预测得分
model.fix_model(intMat, intMat, drugMat, targetMat, seed=[7771, 8367, 22, 1812, 4659])
# x0, y0 为0的下标
x0, y0 = np.where(intMat == 0.0)
print x0, y0
# 处理已标记作用的值
'''
    value:
        'e':    0.000001
        'gpcr': 0.000010
        'nr':   0.100000
        'ic':   0.000001

'''
pr = set_1_litte(intMat, model.predictR, value=0.000010)
# 归一化
pr_norm = norm(pr)
#
scores = re_scores(pr_norm, x0, y0)
# 从大到小排序 返回下标
ii = np.argsort(scores)[::-1]
# 预测数量
predictNum = 10000
# 配对预测药物-标靶对
predict_pairs = [(drug_names[x0[i]], target_names[y0[i]], scores[i]) for i in ii[:predictNum]]

method = 'NetLapRLS'
# 保存新作用对的路径
new_dti_file = os.path.join(output_dir, "_".join([method, dataset, "new_dti.txt"]))
novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'biodb'))
