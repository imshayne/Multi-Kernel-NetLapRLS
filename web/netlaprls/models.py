# _*_ coding: utf-8 _*_

from __future__ import unicode_literals

from django.db import models
# Create your models here.


class Enzyme(models.Model):
    drug_id = models.CharField(max_length=16)
    target_id = models.CharField(max_length=16)
    score = models.FloatField(max_length=16)
    kegg = models.CharField(max_length=1, default='')
    drugbank = models.CharField(max_length=1, default='')
    chembl= models.CharField(max_length=1, default='')
    matador = models.CharField(max_length=1, default='')

    def __str__(self):
        return 'Enzyme'


class Ion_Channel(models.Model):
    drug_id = models.CharField(max_length=16)
    target_id = models.CharField(max_length=16)
    score = models.FloatField(max_length=16)
    kegg = models.CharField(max_length=1, default='')
    drugbank = models.CharField(max_length=1, default='')
    chembl = models.CharField(max_length=1, default='')
    matador = models.CharField(max_length=1, default='')

    def __str__(self):
        return 'Ion_Channel'


class GPCR(models.Model):
    drug_id = models.CharField(max_length=16)
    target_id = models.CharField(max_length=16)
    score = models.FloatField(max_length=16)
    kegg = models.CharField(max_length=1, default='')
    drugbank = models.CharField(max_length=1, default='')
    chembl = models.CharField(max_length=1, default='')
    matador = models.CharField(max_length=1, default='')

    def __str__(self):
        return 'GPCR'


class Nuclear_Receptor(models.Model):
    drug_id = models.CharField(max_length=16)
    target_id = models.CharField(max_length=16)
    score = models.FloatField(max_length=16)
    kegg = models.CharField(max_length=1, default='')
    drugbank = models.CharField(max_length=1, default='')
    chembl = models.CharField(max_length=1, default='')
    matador = models.CharField(max_length=1, default='')

    def __str__(self):
        return 'Nuclear_Receptor'