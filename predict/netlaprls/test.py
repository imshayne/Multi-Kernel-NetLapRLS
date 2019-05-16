# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 10:29'

import numpy as np
from functions import *

# mat = np.arange(6).reshape(2,3)
# inM = np.zeros((2, 3))
#
# inM += 1./4*mat
# print mat
# print inM
#
# print inM.T.shape
data_dir = os.path.abspath('..')
print data_dir

# 取出数据集
dataset = 'e'

# inMat作用矩阵 drugMat药物相似矩阵 targetMat标靶相似矩阵
# intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
# 返回drug target的名称
# drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))
# print drug_names
# print target_names
# print intMat.shape, drugMat.shape, targetMat.shape
#
# wt = np.arange(6).reshape(3,2)
# print np.sum([wt, wt], dtype=np.float64, axis=1)
# print np.linalg()
# 训练集和测试集


## np.diag()
x = np.linspace(1,4,9).reshape((3,3))
x[0, 0:] = 0
print x
xl = np.sum(x, axis=1)
print xl
# print np.diag(1.0/np.sqrt(np.sum(x, axis=1, dtype=np.float64)))
# print np.diag(np.sqrt(1.0/np.sum(x, axis=1, dtype=np.float64)))
xl[np.where(np.sum(x, axis=1)==0)] = 1
xl[np.where(xl==0)] == 1
print xl

x = np.arange(9).reshape(3, 3)
print np.array(x)
print np.array(x).dtype
y = np.array(x)
print y[:, 0], y[:,1]

for i, j in ['tk', 'ik']:
    print i, j
    print "\n"

x = "K=5"
print x.split("=")

temp = 'c=5 K1=5 K2=5 r=100 lambda_d=0.125 lambda_t=0.125 alpha=0.25 beta=0.125 theta=0.5'
for s in str(temp).split():
    print s.split('=')

for key,val in [s.split('=') for s in str(temp).split()]:
    print key, val


############################################################################
# defaultdict 实例
from collections import defaultdict
s = [('Python', 1), ('Swift', 2), ('Python', 3), ('Swift', 4), ('Python', 9)]
# 创建defaultdict，设置由list()函数来生成默认值
d = defaultdict(list)
for k, v in s:
    # 直接访问defaultdict中指定key对应的value即可。
    # 如果该key不存在，defaultdict会自动为该key生成默认值
    d[k].append(v)
print(list(d.items()))
print len(d)

############################################################################