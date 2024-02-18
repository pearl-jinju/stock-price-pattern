from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

from .models import StockPriceDateBase, StockNameAll
import pandas as pd


# Create your views here.
class MainPage(APIView):
    def get(self, request):

        # 검색창을 위한 검색어 db를 메인페이지에 리스트로 넘겨줌

        df_name_ticker = StockNameAll.objects.all()
        df_name_ticker = pd.DataFrame(df_name_ticker.values_list()).iloc[:,1:]
        df_name_ticker.columns = ['종목명','티커']
        df_name_ticker = df_name_ticker.sort_values(by="종목명")
        name_list = df_name_ticker.iloc[:,:1].values.tolist()
        name_list = [name[0] for name in name_list]

        ticker_list = df_name_ticker.iloc[:,1:2].values.tolist()
        ticker_list = [ticker[0] for ticker in ticker_list]

        name_ticker_list = [name+" ("+ticker+")" for name, ticker in zip(name_list,ticker_list)]
        name_ticker_list =",".join(name_ticker_list)
        # print(df_name , df_ticker)


        return render(request,"main/main.html", {'name_ticker_list': name_ticker_list,},status=200) #context html로 넘길것