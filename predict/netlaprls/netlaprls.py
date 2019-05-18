# _*_ coding: utf-8 _*_
__author__ = 'mcy'
__date__ = '2019-05-18 22:11'

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score

'''
Default parameters in netlaprls :
    gamma_d = 10.0
    gamma_t = 10.0
    beta_d = 1e-5
    beta_t = 1e-5
'''


class NetLapRLS:
    def __init__(self, gamma_d=10.0, gamma_t=10.0, beta_d=1e-5, beta_t=1e-5):
        self.gamma_d = float(gamma_d)
        self.gamma_t = float(gamma_t)
        self.beta_d = float(beta_d)
        self.beta_t = float(beta_t)

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W * intMat
        m, n = R.shape
        drugMat = (drugMat + drugMat.T) / 2.
        targetMat = (targetMat + targetMat.T) / 2.
        Wd = (drugMat + self.gamma_d * np.dot(R, R.T)) / (1.0 + self.gamma_d)
        Wt = (targetMat + self.gamma_t * np.dot(R.T, R)) / (1.0 + self.gamma_t)
        Wd = Wd - np.diag(np.diag(Wd))
        Wt = Wt - np.diag(np.diag(Wt))
        Wd_srow = np.sum(Wd, axis=1, dtype=np.float64)
        D = np.diag(1.0 / np.sqrt(Wd_srow))
        Ld = np.eye(m) - np.dot(np.dot(D, Wd), D)
        Wt_srow = np.sum(Wt, axis=1, dtype=np.float64)
        D = np.diag(1.0 / np.sqrt(Wt_srow))
        Lt = np.eye(n) - np.dot(np.dot(D, Wt), D)
        X = np.linalg.inv(Wd + self.beta_d * np.dot(Ld, Wd))
        Fd = np.dot(np.dot(Wd, X), R)
        X = np.linalg.inv(Wt + self.beta_t * np.dot(Lt, Wt))
        Ft = np.dot(np.dot(Wt, X), R.T)
        self.predictR = 0.5 * (Fd + Ft.T)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
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
