# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 15:10'

from functions import *
from netlaprls1_0rev import *
import  numpy as np
import os
import pickle

# 读路径
data_dir = os.path.abspath('..')
print data_dir

# 取出数据集
dataset = 'e'

# inMat作用矩阵 drugMat药物相似矩阵 targetMat标靶相似矩阵
intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
# 返回drug target的名称
drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))

# print np.argwhere(np.isnan(targetMat))
# print intMat, drugMat, targetMat
# 数据开始fit
nlp = NetLapRLS()
nlp.fix_model(1, intMat, drugMat, targetMat)

print nlp.predictR
# print intMat.shape == nlp.predictR.shape
# print intMat.shape
# print len(drug_names)
# # print nlp.predictR
# # test_data = np.arange(9).reshape((3,3))
# #
# # scores = nlp.predict_scores(test_data, 0)
# # print scores
# #
# # intMat_Test = intMat.astype(np.int)
# # intMat_Value = nlp.predict_scores(intMat_Test,0)
# #
# # nlp.evaluation(intMat, intMat_Value)
# seeds = [22]
# cv_data = cross_validation(intMat, seeds)
# print cv_data
# print '********'*10
#
# print type(cv_data)

print np.logspace(0,-9,10)