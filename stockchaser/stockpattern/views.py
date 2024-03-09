from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

# from loader import loaddata

from .models import StockPriceDateBase,StockNameAll, StockPricePattern
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
        # ==================================================
        # 주가 정보를 넘겨줌 from stockPrice pattern
        # df = pd.DataFrame(StockPricePattern.objects.filter(name="AJ네트웍스").values_list())
        # df.columns = ['id','종목명','분석기간','시작일자','종료일자','주가_list','등락률_list','MA224_list','주가/MA224_list','rate_MA224_mean_list','rate_MA224_std_list','yield_5days','yield_20days','yield_60days']
        # df = df.sort_values(by="종료일자",ascending=True)
        
        # price_list  = df['주가_list'].iloc[0]
        # start_date  = df['시작일자'].iloc[0]
        # end_date  = df['종료일자'].iloc[0]
        # date_list =[]
        # for i in range(244):
        #     if i == 0:
        #         date_list.append(start_date)
        #     else:
        #         date_list.append("")
        # date_list[-1] = end_date
        # ==================================================
        df = pd.DataFrame(StockPriceDateBase.objects.filter(name="AJ네트웍스").values_list())
        df.columns = ['id','날짜','종목명','티커','시가','고가','저가','종가','등락률','거래량']
        df = df.sort_values(by="날짜",ascending=True)
        

        # 넘겨줄 데이터 완성
        price_list  = df['종가'].values.tolist()[-25:]
        date_list  = df['날짜'].values.tolist()[-25:]




        return render(request,"main/main.html", {
            'name_ticker_list': name_ticker_list,
            'price_list': price_list,
            'date_list': date_list,
            },status=200) #context html로 넘길것
        # return render(request,"main/main.html",status=200)