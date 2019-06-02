# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-06-02 14:15'

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
from netlaprls.models import New_pairs


def remove():
    zero_pairs = New_pairs.objects.filter(score=0).delete()
    print zero_pairs

remove()

# (46, {u'netlaprls.New_pairs': 46})