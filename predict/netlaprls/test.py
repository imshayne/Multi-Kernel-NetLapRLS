# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 10:29'

from trainmodel import TrainModel
from cv_eval import netlaprls_cv_eval

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
trainmodel = TrainModel(dataset='ic')
trainmodel.train_md()

# 调参 beta_t bata_d
trainmodel.cv_eval()
