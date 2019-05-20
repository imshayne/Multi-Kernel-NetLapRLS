# _*_ coding: utf-8 _*_

from django.shortcuts import render
from django.views.generic.base import View
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
# Create your views here.


def test(request):
    return render(request, 'index.html')