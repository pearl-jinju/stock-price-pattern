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
    # 계속데이터가 중복되는경우에 정지하는 카운터를 초기화
    stop_cnt = 0
    for i in tqdm(range(start,end)): 
        if stop_cnt > 2:
            print("데이터 입력을 조기 종료합니다")
            break
        # 영업일자와 당일의 영업일자가 같은경우 패스
        if i>0 and date_from_now(i) == date_from_now(i-1):
            print("pass")
            pass
        else:
            df_result = get_stock_basic_info(i)

            # 만약 동일 날짜의 데이터가 있다면?
            if StockPriceDateBase.objects.filter(date=df_result['일자'].iloc[0]).count()>0:
                stop_cnt += 1
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


# 주가 패턴을 만듬
class CreatePattern(APIView):
    def get(self, request):

        # 상장한지 1년이 안된 신규상장주는 분석기능 미제공 


        # 종목명+ 티커 형태의 df를 불러옴
        df_name_ticker = StockNameAll.objects.all()
        df_name_ticker = pd.DataFrame(df_name_ticker.values_list()).iloc[:,1:]
        df_name_ticker.columns = ['종목명','티커']


        # 주가 데이터를 모두 가져옴
        StockPriceDB_df = StockPriceDateBase.objects.all()
        StockPriceDB_df = pd.DataFrame(StockPriceDB_df.values_list()).iloc[:,1:]
        StockPriceDB_df.columns = ['날짜','종목명','티커','시가','고가','저가','종가','등락률','거래량','거래대금','PER','BPS','PBR','EPS','DIV','DPS']

        # 주가의 6개월 패턴을 분해한다. [날짜, 종목명, 종가, 등락률]

        # 날짜순 정렬
        StockPriceDB_df_stocks = StockPriceDB_df[StockPriceDB_df['종목명']=="더존비즈온"]
        StockPriceDB_df_stocks = StockPriceDB_df_stocks.sort_values(by="날짜", ascending=True)
        # print(StockPriceDB_df_stocks)
                
        # 종가데이터만을 받음
        StockPriceDB_df_stocks = StockPriceDB_df_stocks[['날짜', '종목명', '종가', '등락률']]
        # 224일 평균선
        StockPriceDB_df_stocks['MA224'] = StockPriceDB_df_stocks['종가'].rolling(window=224).mean()
        # 주가 / 224일 평균
        StockPriceDB_df_stocks['MA224_rate'] = StockPriceDB_df_stocks['종가']/StockPriceDB_df_stocks['MA224']

        # 날짜기준으로 각각 pivoting
        # 주가기준
        StockPriceDB_df_stocks_price = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="종가")
        # # 등락률 기준
        StockPriceDB_df_stocks_rate = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="등락률")
        # # # MA224 기준
        # StockPriceDB_df_stocks_MA224 = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224")
        # # 주가/MA224 기준
        StockPriceDB_df_stocks_MA224_rate = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224_rate")

        # DateFrame 초기화
        pivotted_StockPriceDB_df_stocks_data_result = pd.DataFrame()

        for idx in tqdm(range(len(StockPriceDB_df_stocks)-120)):
            # ------------------기초데이터 추출
            stock_name = StockPriceDB_df_stocks_price.iloc[:,idx:idx+121].index[0]
            # 컬럼 첫번째 날짜 
            start_day = StockPriceDB_df_stocks_price.iloc[:,idx:idx+121].columns[1]
            # 컬럼 마지막 날짜
            last_day = StockPriceDB_df_stocks_price.iloc[:,idx:idx+121].columns[-1]
            # 구분자로 --를 사용
            name_date = stock_name +"--"+start_day+"--"+last_day

            # ------------------ 데이터 제작
            temp_data_df = pd.DataFrame()

            # 주가 결과 데이터
            temp_df_price = StockPriceDB_df_stocks_price.iloc[:,idx:idx+121].copy().values.tolist()
            # 등락률 결과 데이터
            temp_df_rate = StockPriceDB_df_stocks_rate.iloc[:,idx:idx+121].copy().values.tolist()
            # 주가/MA224  결과 데이터
            temp_df_MA224_rate = StockPriceDB_df_stocks_MA224_rate.iloc[:,idx:idx+121].copy().values.tolist()


            temp_data_df['종목명'] = stock_name
            temp_data_df['시작일자'] = start_day
            temp_data_df['종료일자'] = last_day
            temp_data_df['주가_list'] = temp_df_price
            temp_data_df['등락률_list'] = temp_df_rate
            temp_data_df['주가/MA224_list'] = temp_df_MA224_rate

        #     # ------------------데이터프레임에 추가
            pivotted_StockPriceDB_df_stocks_data_result = pd.concat([pivotted_StockPriceDB_df_stocks_data_result,temp_data_df],axis=0)


        # 종목명 별로 날짜기준 정렬함
        print(pivotted_StockPriceDB_df_stocks_data_result)
        # # ------------------- DB에 저장(중복검사) 

        return Response(status=200)

class LoadPrice(APIView):
    def get(self, request):
        #반복하여 데이터에 집어넣음
        # 당일의 주가동향은 바뀌므로 저장하지 않고, 전일 데이터까지만 저장함
        loaddata(1,100)
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