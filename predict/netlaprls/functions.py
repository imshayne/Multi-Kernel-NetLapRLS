# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 10:19'

import numpy as np
import os
from collections import defaultdict



'''
此文件中为初始化操作数据集加载(load_data_from_file)、矩阵行列名称的读取(get_drugs_targets_names)
'''


def load_data_from_file(dataset, folder):
    # 读取药物-标靶相互作用矩阵 并初始化返回
    MatShape = None
    with open(os.path.join(folder, dataset+'_admat_dgc.txt'), "r")as inf:
        inf.next()
        int_array = [line.strip('\n').split()[1:] for line in inf]
        intMat = np.array(int_array, dtype=np.float64).T
        MatShape = intMat.shape

    # 读取 行列的长度 初始化相似矩阵
    drug_num, target_num = MatShape
    drugMat = np.zeros((drug_num, drug_num), dtype=np.float64)
    targetMat = np.zeros((target_num, target_num), dtype=np.float64)

    # 读取药物相似矩阵 包括4个核矩阵 并且分别乘以1/4 合并为新的药物相似矩阵
    for indexD in xrange(1, 5, 1):
        with open(os.path.join(folder, dataset+"_simmat_dc%d.txt" % indexD), "r") as inf:
            inf.next()
            int_array = [line.strip("\n").split()[1:] for line in inf]
            intSimmat = np.array(int_array, dtype=np.float64)
            # print intSimmat
            # 使用np.sum 函数求平均矩阵
            drugMat = np.sum([0.2*intSimmat, drugMat], dtype=np.float64, axis=0)
            # drugMat += 0.25*intSimmat + drugMat
    # 读取标靶相似矩阵 包括4个核矩阵 并且分别乘以1/4 合并为新的标靶相似矩阵
    for indexT in xrange(1, 5, 1):
        with open(os.path.join(folder, dataset+"_simmat_dg%d.txt" % indexT), "r") as inf:
            inf.next()
            int_array = [line.strip("\n").split()[1:] for line in inf]
            intSimmatS = np.array(int_array, dtype=np.float64)
            targetMat = np.sum([0.2*intSimmatS, targetMat], dtype=np.float64, axis=0)
            # targetMat += 0.25*intSimmatS + targetMat
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        drugs = inf.next().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    # print cv_data
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size/num
        for i in xrange(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


# 训练模型
def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


# 平均置信区间
def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    # sem 计算标准误差
    m, se = np.mean(a), scipy.stats.sem(a)
    # ppf 分位点函数
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)
