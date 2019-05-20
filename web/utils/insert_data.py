# !/usr/bin/env python
# coding:utf-8

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")

'''
Django 版本大于等于1.7的时候，需要加上下面两句
import django
django.setup()
否则会抛出错误 django.core.exceptions.AppRegistryNotReady: Models aren't loaded yet.
'''

import django

if django.VERSION >= (1, 7):  # 自动判断版本
    django.setup()

BASE_DIR = os.path.abspath('../predict_dataset')
'''
NetLapRLS_e_new_dti.txt
NetLapRLS_gpcr_new_dti.txt
NetLapRLS_ic_new_dti.txt
NetLapRLS_nr_new_dti.txt
'''

datasets = ['e', 'gpcr', 'ic', 'nr']


def main():
    from netlaprls.models import Enzyme, GPCR, Ion_Channel, Nuclear_Receptor
    for dataset in datasets:
        f = open(os.path.join(BASE_DIR, 'NetLapRLS_' + dataset + '_new_dti.txt'))
        for line in f:
            drug_id, target_id, score, chembl ,drugbank, kegg, matador = line.split('\t\t')[1:]
            if 'e' == dataset:
                Enzyme.objects.create(drug_id=drug_id, target_id=target_id, score=score, kegg=kegg, drugbank=drugbank, chembl=chembl, matador=matador)
            elif 'gpcr' == dataset:
                GPCR.objects.create(drug_id=drug_id, target_id=target_id, score=score, kegg=kegg,
                                                drugbank=drugbank, chembl=chembl, matador=matador)
            elif 'ic' == dataset:
                Ion_Channel.objects.create(drug_id=drug_id, target_id=target_id, score=score, kegg=kegg,
                                            drugbank=drugbank, chembl=chembl, matador=matador)
            elif 'nr' == dataset:
                Nuclear_Receptor.objects.create(drug_id=drug_id, target_id=target_id, score=score, kegg=kegg,
                                            drugbank=drugbank, chembl=chembl, matador=matador)
        f.close()


if __name__ == "__main__":
    main()
    print('Done!')