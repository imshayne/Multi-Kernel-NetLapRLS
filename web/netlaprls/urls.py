# _*_coding:utf-8_*_
__author__ = 'mcy'
__date__ = '2019-05-22 09:13'


from django.conf.urls import url
from . import views
urlpatterns = [

    url(r'^$', views.netlaprls, name='netlaprls'),
    url(r'^search-pairs/$', views.search_pairs, name='search_pairs'),

]