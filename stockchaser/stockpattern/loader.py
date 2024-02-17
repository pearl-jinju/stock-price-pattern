from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

from sqlalchemy import create_engine
import pandas as pd
import pymysql
import time


from .models import StockPriceDateBase, StockNameAll#, StockPriceDB


# 필요 모듈
from pykrx import stock
import pandas as pd
import datetime
from tqdm import tqdm
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import pickle
import os

# 시간 설정 함수
def date_from_now(day:int=0, type="L"):
    '''
    영업일 기준 최근일자 호출함수(L= linked_date_type ex- %Y%m%d , S=seperate_date_type ex- %Y-%m-%d)\n
    date_from_now() 는 오늘일자 date_from_now(365)은 365일 전 일자를 str형태로 출력한다.
    '''        
    edate_str = stock.get_nearest_business_day_in_a_week()
    if type == "L":
        edate_dt = datetime.datetime.strptime(edate_str, '%Y%m%d')
        sdate_dt = edate_dt-datetime.timedelta(days=day)
        sdate_str = str(sdate_dt)[:4]+str(sdate_dt)[5:7]+str(sdate_dt)[8:10]
        # print(sdate_str)

    elif type == "S":
        edate_dt = datetime.datetime.strptime(edate_str, '%Y%m%d')
        sdate_dt = edate_dt-datetime.timedelta(days=day)
        sdate_str = sdate_dt.strftime('%Y-%m-%d')

    return_date = stock.get_nearest_business_day_in_a_week(sdate_str)
    return return_date



# 종목 관련 함수
def get_stock_basic_info(day=0, market="ALL", detail="ALL"):
    """ 종목 기초정보 제공 함수
        티커/종목명/시가/종가/변동폭(오늘)/등락률(오늘)/\n
        거래량(오늘)/거래대금(오늘)/상장주식수/보유수량/\n
        지분율/한도수량/한도소진률/BPS/PER/PBR/EPS/DIV/DPS_
        
        All = 모든 정보
        BASIC = 기초 정보

    Args:
        day (int, optional): _description_. Defaults to 0.
        market (str, optional): _description_. Defaults to "ALL".
        detail (str, optional): _description_. Defaults to "ALL".

    Returns:
        _type_: DataFrame
    """
    if detail=="ALL":
        # day일전(영업일기준) 일자를 불러옴
        df_name = stock.get_market_price_change(date_from_now(day),date_from_now(day),market="ALL").reset_index()[['종목명','티커']]
        df_basic = stock.get_market_ohlcv(date_from_now(day), market=market).reset_index()
        df_fundamental = stock.get_market_fundamental( date_from_now(day), market=market).reset_index()
        df_result = pd.merge(df_basic,df_fundamental, on='티커',how='left')
        df_result = pd.merge(df_result,df_name, on='티커',how='left')
        # 일자의 문자화
        str_date = str(date_from_now(day))
        df_result.loc[:, '일자'] = str_date[0:4]+"-"+str_date[4:6]+"-"+str_date[6:]
        df_result['티커'] = df_result['티커'].astype(dtype='object')
        df_result =df_result[['일자','종목명','티커','시가','고가','저가','종가','등락률','거래량','거래대금','PER','BPS','PBR','EPS','DIV','DPS']]     
        df_result = df_result.where(pd.notnull(df_result), None)
        return df_result
    if detail=="BASIC":
        df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
        return df_change


def loaddata(start,end):


    for i in tqdm(range(start,end)): 
        # 영업일자와 당일의 영업일자가 같은경우 패스
        if i>0 and date_from_now(i) == date_from_now(i-1):
            print("pass")
            pass
        else:
            df_result = get_stock_basic_info(i)

            # 만약 동일 날짜의 데이터가 있다면?
            if StockPriceDateBase.objects.filter(date=df_result['일자'].iloc[0]).count()>0:
                print("데이터가 이미 있습니다.")
                pass
            else:
                df_bulk =[]
                for i in tqdm(range(len(df_result))):
                    df_bulk.append(
                        StockPriceDateBase(
                        date             = df_result['일자'].iloc[i],
                        name             = df_result['종목명'].iloc[i],   # 종목명
                        ticker           = df_result['티커'].iloc[i],  # 티커
                        start_price      = df_result['시가'].iloc[i],   # 시가
                        high_price       = df_result['고가'].iloc[i],   # 고가
                        low_price        = df_result['저가'].iloc[i],   # 저가
                        end_price        = df_result['종가'].iloc[i],   # 종가
                        fluctuation_rate = df_result['등락률'].iloc[i],   # 등락률
                        volume           = df_result['거래량'].iloc[i],   # 거래량
                        volume_amount    = df_result['거래대금'].iloc[i], # 거래대금
                        # -값이 나올수 있는 per 때문에 지표들은 모두 text 처리를 우선으로 함
                        per              = str(df_result['PER'].iloc[i]),   # per
                        bps              = str(df_result['BPS'].iloc[i]), # bps
                        pbr              = str(df_result['PBR'].iloc[i]),   # pbr
                        eps              = str(df_result['EPS'].iloc[i]),   # eps
                        div              = str(df_result['DIV'].iloc[i]),  # div
                        dps              = str(df_result['DPS'].iloc[i]),  # dps
                        )
                    )

                StockPriceDateBase.objects.bulk_create(df_bulk,ignore_conflicts=True)

    # 종목명 DB를 갱신함
    name_df = StockPriceDateBase.objects.all()
    name_df = pd.DataFrame(name_df.values_list()).iloc[:,2:4]
    name_df.columns = ["종목명","티커"]
    # 중복 제거
    name_df.drop_duplicates(keep="last", inplace=True)
    print(name_df)

    # 기존 DB 제거
    StockNameAll.objects.all().delete()


    name_df_bulk =[]
    for i in tqdm(range(len(name_df))):
        name_df_bulk.append(
            StockNameAll(
        name             = name_df['종목명'].iloc[i],   # 종목명
        ticker           = name_df['티커'].iloc[i],  # 티커
                                        )
        )
    StockNameAll.objects.bulk_create(name_df_bulk,ignore_conflicts=True)
    print("종목명 갱신완료")


# 부족한 일자만큼만 채움
class FillLoadPrice(APIView):
    def get(self, request):
        # sql에 저장된 db의 유니크 값을 가져옴

        # date_from_now를 이용해 최근 100일의 날짜를 모두 리스트에 저장함

        #  두집합에서 부족한 일수만큼만 반복하여 데이터를 집어넣음

        return Response(status=200)

class LoadRecentPrice(APIView):
    def get(self, request):
        # 로더는 시간이 너무 오래걸리므로 -1영업일의 데이터만을 db에 저장해야함
        # DB에 저장후에는 없는 날짜만 채워서 데이터를 보충하는 일이 필요함
        # 최근일자의 주가 정보는 확정되지 않은 값이므로 전날일자만 저장
        df = StockPriceDateBase.objects.all()
        df = pd.DataFrame(df.values_list())
        df = df.loc[:,1:1] #.sort_values(by="1",ascending=False)
        df.columns = ['일자']
        # 50일 전의 일자까지 겹치는지 조회
        df_list = pd.DataFrame(df['일자']).sort_values(by="일자", ascending=False)['일자'].unique()[1:50]

        # 현재 일자로 부터 1일전 영업일부터 영업일 일자 뽑기
        real_date_list = []
        for i in range(1,50):
            date = str(date_from_now(i))
            yyyy = date[0:4]
            mm = date[4:6]
            dd = date[6:8]
            final_date = yyyy+"-"+mm+"-"+dd
            if final_date not in real_date_list:
                real_date_list.append(final_date)

        recent_date_list = []
        for real, idx in zip(real_date_list,range(1,len(real_date_list)+1)):
            if real not in df_list:
                recent_date_list.append(idx)

        for i in recent_date_list:
            loaddata(i,i+1)
        return Response(status=200)

class LoadPrice(APIView):
    def get(self, request):
        #반복하여 데이터에 집어넣음
        # TODO 데이터 순서 검증을 거꾸로 하여 데이터 누락분 확인 필요
        # loaddata(708,10000)
        # loaddata(2319,10000)
        loaddata(1,5)
        return Response(status=200)


class DeletePrice(APIView):
    def get(self, request):
        # 데이터 삭제  
        StockPriceDateBase.objects.all().delete()

        # StockPriceDateBase.objects.bulk_create
        # YourModel.objects.bulk_create([
        # YourModel(column1=row['column1'], column2=row['column2']) for _, row in data.iterrows()
        # ])
        print(StockPriceDateBase.objects.count())   
        return Response(status=200)

# class DBcheck(APIView):
#     def get(self, request):
#         # DB연결 확인
#         db_connection_path = 'mysql+pymysql://y2kwlswn:wjd7615@localhost:8000/StockPriceDB'
#         db_connection = create_engine(db_connection_path)

#         conn = db_connection.connect()
#         return Response(status=200)