# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 10:19'

import numpy as np
import os
from collections import defaultdict

'''
此文件中为初始化操作数据集加载(load_data_from_file)、矩阵行列名称的读取(get_drugs_targets_names)
'''

BASE_DIR = os.path.abspath('..')
BASE_DIR = os.path.join(BASE_DIR, 'output')


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

    # 读取药物相似矩阵 包括4个核矩阵 并且分别乘以0.2 合并为新的药物相似矩阵 (剩下的0.2作为正则项)
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


# cv的模式cv1 cv2 cv3
# cv1 在intMat的基础上，seeds = [7771, 8367, 22, 1812, 4659]
def cross_validation(intMat, seeds, cv=1, num=10):
    cv_data = defaultdict(list)
    # seeds 随机数种子
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:  # cvs1 cvs2
            index = prng.permutation(num_drugs)
        if cv == 1:
            # index 表示intMat中全体下标总数
            # permutation test 置换检验
            index = prng.permutation(intMat.size)
        step = index.size/num
        for i in xrange(num):
            if i < num-1:
                # 抽取step步长的下标 测试数据的长度
                ii = index[i*step:(i+1)*step]
            else:
                # i = 9
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            # 生成随机test_data对应intMat下的下标
            x, y = test_data[:, 0], test_data[:, 1]
            # 找出对应intMat下标的值
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            # 将上述找出的下标所对应的值 置为0 W即作为训练矩阵的标记矩阵
            W[x, y] = 0
            # 将生成的训练矩阵 结合上述标记为0的下标 和 test_label(没置为0前对应的值)
            cv_data[seed].append((W, test_data, test_label))
    return cv_data  # 每次seed内循环生成10个训练集


# 训练模型
def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    # 保存testdata testlabel
    test_scores = []
    test_labels = []
    # cv_data.keys()返回 cross_validation()中 列表seeds
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            # 将10折测试集 分别进行模型评估
            model.fix_model(W, intMat, drugMat, targetMat, seed)   # 得到 intMat-->Ytree 的 predictR矩阵
            # scores test_label 保存 为了plot 出 aupr 和 pr 图
            aupr_val, auc_val, score, test_label = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
            # print test_data
            # print test_label
            test_scores.append(score)
            test_labels.append(test_label)  #
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64) , test_scores, test_labels


def train_accuracy(model, cv_data, intMat, drugMat, targetMat):
    accs = []
    # cv_data.keys()返回 cross_validation()中 列表seeds
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            # 将10折测试集 分别进行模型评估
            model.fix_model(W, intMat, drugMat, targetMat, seed)  # 得到 intMat-->Ytree 的 predictR矩阵
            acc = model.accuracy_md(test_data, test_label)  #
            accs.append(acc)
    return np.array(accs, dtype=np.float64)


# aupr平均值 平均置信区间  TODO 置信区间 由样本统计量所构造的总体参数的估计区间
# 样本量相同的情况下， 置信水平越高 置信区间越宽
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
    np.savetxt(file_name, auc_vec, fmt='%s')


# 多变量写入文件
def write_multi_para_to_file(file_name, *para):
    file_name = os.path.join(BASE_DIR, file_name)
    np.savez(file_name, para)


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


# 多变量文件读写
def load_multi_para_to_file(file_name):
    file_name = os.path.join(BASE_DIR, file_name)
    return np.load(file_name)


# 矩阵保存
def save_dt_matrix_to_file(filename, matrix):
    filename = os.path.join(BASE_DIR, filename)
    np.savetxt(filename, matrix, delimiter=',', fmt='%.8f')


def load_dt_matrix_to_file(filename):
    filename = os.path.join(BASE_DIR, filename)
    return np.loadtxt(filename, delimiter=',')
