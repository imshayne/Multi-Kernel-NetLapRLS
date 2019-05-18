# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-04-29 15:06'

'''
netlaprls
'''

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, precision_score
from sklearn.metrics import auc

'''
Default parameters in [1]:
    gamma_d1 = gamma_p1 = 1
    gamma_d2 = gamma_p2 = 0.01
    beta_d = beta_p = 0.3
Default parameters in this implementation:
    gamma_d = 0.01, gamma_d=gamma_d2/gamma_d1
    gamma_t = 0.01, gamma_t=gamma_p2/gamma_p1
    beta_d = 0.3
    beta_t = 0.
'''


class NetLapRLS:
    def __init__(self, gamma_d=10.0, gamma_t=10.0, beta_d=1e-5, beta_t=1e-4):
        self.gamma_d = float(gamma_d)
        self.gamma_t = float(gamma_t)
        self.beta_d = float(beta_d)
        self.beta_t = float(beta_t)

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        m, n = R.shape
        # 修正similarity matrix --> symmetric matrix
        drugMat = (drugMat + drugMat.T)/2.
        targetMat = (targetMat + targetMat.T)/2.

        # 药物相似度矩阵
        Wd = (drugMat+self.gamma_d*np.dot(R, R.T)) / (1.0+self.gamma_d)
        Wt = (targetMat+self.gamma_t*np.dot(R.T, R)) / (1.0+self.gamma_t)
        # 去除主对角线上自身的评分e
        Wd = Wd-np.diag(np.diag(Wd))
        Wt = Wt-np.diag(np.diag(Wt))
        # D是一个wd的按照列相加 得到的为列表 长度为drug 开根号 的对角矩阵  D=Dd^(-1/2)
        ## Dd节点的度矩阵
        Wd_srow = np.sum(Wd, axis=1, dtype=np.float64)
        D = np.diag(1.0/np.sqrt(Wd_srow))
        Ld = np.eye(m) - np.dot(np.dot(D, Wd), D)  # Ld = Indxnd - DwdD

        Wt_srow = np.sum(Wt, axis=1, dtype=np.float64)
        # print np.where(Wt_srow == 0)  # 检查np.sum(Wt,axis=1))是否有0
        D = np.diag(1.0/np.sqrt(Wt_srow))
        Lt = np.eye(n) - np.dot(np.dot(D, Wt), D)

        # np.linalg为行业标准级fortran库  inv求矩阵的逆  X = (wd + beta_d*wd)^(-1)
        X = np.linalg.inv(Wd+self.beta_d*np.dot(Ld, Wd))
        Fd = np.dot(np.dot(Wd, X), R)
        X = np.linalg.inv(Wt+self.beta_t*np.dot(Lt, Wt))
        Ft = np.dot(np.dot(Wt, X), R.T)
        self.predictR = 0.5*(Fd+Ft.T)

    # 归一化 预测矩阵
    def normalize_md(self, F):
        max = np.max(F)
        min = np.min(F)
        print 'max min'
        print max, min
        return ((f - min)/(max - min) for f in F)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        # TODO 输出预测值之前正则化
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        # test_label的值为原始intMat中对应下标的值 scores是将原始intMat中对应test_data的下标置为0后经过fix后 对应下标位置的得分
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def accuracy_md(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        return accuracy_score(test_label, scores.round(), normalize=True)

    def __str__(self):
        return "Model: NetLapRLS, gamma_d:%s, gamma_t:%s, beta_d:%s, beta_t:%s" \
               % (self.gamma_d, self.gamma_t, self.beta_d, self.beta_t)