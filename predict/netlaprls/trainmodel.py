# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-15 10:09'
from cv_eval import *
'''
    :param dataset = 'e' 'gpcr' 'ic' 'nr'
    :param cv = 0 | 1    cvs = 1,2,3
    :param seeds = [7771, 8367, 22, 1812, 4659]
'''


class TrainModel:
    def __init__(self, dataset='e', cvs=1, seeds=[22, ], output_dir='output'):
        self.dataset = dataset
        self.seeds = seeds
        self.cvs = cvs
        self.data_dir = os.path.abspath('..')
        self.output_dir = output_dir
        if (self.cvs == 2) or (self.cvs == 3):
            self.cv = 0
        else:
            self.cv = 1

    # 读取并且保存已读到的矩阵
    def read_and_write_datasets(self):
        intMat, drugMat, targetMat = load_data_from_file(self.dataset, os.path.join(self.data_dir, 'datasets'))
        try:
            save_dt_matrix_to_file('intMat_'+self.dataset, intMat)
            save_dt_matrix_to_file('drugMat_'+self.dataset, drugMat)
            save_dt_matrix_to_file('targetMat_'+self.dataset, targetMat)
        finally:
            print '%s dt_matrix save done! \n' % self.dataset

        return intMat, drugMat, targetMat

    # 读取并且保存已读到的行列名
    def read_and_write_dg_names(self):
        drug_names, target_names = get_drugs_targets_names(self.dataset, os.path.join(self.data_dir, 'datasets'))
        try:
            write_multi_para_to_file('dt_name_'+self.dataset, drug_names, target_names)
        finally:
            print '%s dt_name save done! \n' % self.dataset
        return drug_names, target_names

    def train_md(self):
        self.intMat, self.drugMat, self.targetMat = self.read_and_write_datasets()
        drug_names, target_names = self.read_and_write_dg_names()
        # 使用默认参数fix
        model = NetLapRLS()
        # 交叉验证
        self.cv_data = cross_validation(self.intMat, self.seeds, self.cv)
        tic = time.clock()
        aupr_vec, auc_vec = train(model, self.cv_data, self.intMat, self.drugMat, self.targetMat)
        # 通过aupr_vec 得出平均aupr 和
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print "auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f \n" \
              % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)
        try:
            write_metric_vector_to_file(auc_vec, os.path.join(os.path.join(self.data_dir, self.output_dir), \
                                                              'netlaprls' + "_auc_cvs" + str(self.cvs) + "_" + self.dataset + ".txt"))
            write_metric_vector_to_file(aupr_vec, os.path.join(os.path.join(self.data_dir, self.output_dir), \
                                                               'netlaprls' + "_aupr_cvs" + str(self.cvs) + "_" + self.dataset + ".txt"))
        finally:
            print 'write_metric_vector_to_file success '

    # TODO args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    def cv_eval(self):
        args = None
        netlaprls_cv_eval('netlaprls', self.dataset, self.cv_data, self.intMat, self.drugMat, self.targetMat, self.cvs, args)
