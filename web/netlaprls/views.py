# _*_ coding: utf-8 _*_
from django.core import serializers
from django.shortcuts import render
from django.views.generic.base import View
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse


# Create your views here.


def netlaprls(request):

    return render(request, 'index.html')


from .models import New_pairs
import json
def search_pairs(request):
    if request.method == "GET":
        id = str(request.GET.get('id')).upper()
        new_pairs = None
        if str(id)[0] == 'D':
            new_pairs = New_pairs.objects.filter(drug_id=str(id)).order_by('-score')
        elif str(id)[0] == 'H':
            new_pairs = New_pairs.objects.filter(target_id=str(id)).order_by('-score')
        #
        # predictNum = 100
        # pairs_list = []
        # predict_pairs = new_pairs[:predictNum]
        # for predict_pair in predict_pairs:
        #     pairs_list.append([predict_pair.drug_id, predict_pair.target_id, predict_pair.score, predict_pair.kegg,
        #                        predict_pair.drugbank, predict_pair.chembl, predict_pair.matador])
        # print pairs_list
        data = {}
        data['data'] = json.loads(serializers.serialize("json", new_pairs))
    print data

    return JsonResponse(data, content_type='application/json')
