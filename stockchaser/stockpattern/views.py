from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

from .models import StockPriceDateBase
import pandas as pd


# Create your views here.
class MainPage(APIView):
    def get(self, request):

        # df =  pd.DataFrame(StockPriceDateBase.objects.filter(name="더존비즈온").values_list())
        # print(df)
        # df = pd.DataFrame(list(df.values()))
        # df = df.sort_values(by="date",ascending=True)
        # print(df)
        # return render(request,"main/main.html",context=dict(datafeed=df)) #context html로 넘길것
        return render(request,"main/main.html") #context html로 넘길것