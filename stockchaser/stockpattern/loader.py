from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from datetime import datetime, timedelta

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pymysql
import time
from ast import literal_eval


from .models import StockPriceDateBase, StockPricePattern, StockNameAll, BusinessDayDate


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


# 코사인 유사도 계산 함수
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

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

    # 244일의 데이터를 먼저 만들것
    # 특정 날짜가 입력되었다면, 특정날짜로 부터 244일 이전의 start_date를 구할것
    # 입력되지 않았다면, 현재 날짜를 기준으로 244일 이전의 start_date를 구할것

# 영업일 데이터를 빠르게 부르기 위해 df를 db에 저장
def create_analysis_date(date=date_from_now(0)):

    # BusinessDayDate.objects.all().delete()

    # 1990년 상장된 삼성전자 주식을 이용하여 날짜리스트를 뽑음
    df = stock.get_market_ohlcv("19800101", date_from_now(0), "005930").reset_index()
    df['날짜'] = df['날짜'].apply(lambda x: str(x)[:4]+str(x)[5:7]+str(x)[8:10])
    date_list = df['날짜'].tolist()

    if datetime.datetime.today().hour>=16:
        date_list = date_list
    else:
        date_list = date_list[:-1]



    if BusinessDayDate.objects.all().count() == 0:
        bulk_date_list =[]
        for i in tqdm(range(len(date_list))):
            bulk_date_list.append(
            BusinessDayDate(
            date             = date_list[i],   # 종목명
            )
        )
        BusinessDayDate.objects.bulk_create(bulk_date_list,ignore_conflicts=True)
    
    else:
        db_date_list = pd.DataFrame((BusinessDayDate.objects.all()).values_list()).iloc[:,1].tolist()

        for_save_date_list = []
        for date in date_list:
            if date not in db_date_list:
                for_save_date_list.insert(0,date)

        bulk_date_list =[]
        for i in tqdm(range(len(for_save_date_list))):
            bulk_date_list.append(
                BusinessDayDate(
                date             = for_save_date_list[i],   # 종목명
                )
            )
        bulk_date_list = reversed(bulk_date_list)
        BusinessDayDate.objects.bulk_create(bulk_date_list,ignore_conflicts=True)
        print("일자 갱신 완료")


    return 

class LoadRecentData(APIView):
    def get(self, request):

        # 영업일을 갱신한다
        create_analysis_date()
        # 기간을 먼저 정하자
        # 1. 오늘이 몇일인지 DB에 현재 날짜(전일가격까지만)가 어디까지 저장되어있는지 확인할것
        # 2. 오늘의 날짜와 현재날짜가 같다면 종료할것
        yesterday = date_from_now(1)
        
        # 당일 기준으로 티커를 모두 가져온다(상장폐지된 주식은 반영하지 않음)
        tickers_kospi = stock.get_market_ticker_list(yesterday, market="KOSPI")
        tickers_kosdaq = stock.get_market_ticker_list(yesterday, market="KOSDAQ")
        tickers_all = tickers_kospi+tickers_kosdaq
        tickers_all = tickers_all[:30]
        
        # 종목명 추출을 위한 data조회
        df_for_name_kospi = stock.get_market_price_change(yesterday, yesterday, market="KOSPI")
        df_for_name_kosdaq = stock.get_market_price_change(yesterday, yesterday, market="KOSDAQ")

        df_for_name = pd.concat([df_for_name_kospi,df_for_name_kosdaq],axis=0)

        df_for_name_ticker = df_for_name['종목명'].reset_index()


        # 종목명 및 티커 최신화
        StockNameAll.objects.all().delete()

        name_df_bulk =[]
        for i in tqdm(range(len(df_for_name_ticker))):
            name_df_bulk.append(
                StockNameAll(
                name             = df_for_name_ticker['종목명'].iloc[i],   # 종목명
                ticker           = df_for_name_ticker['티커'].iloc[i],  # 티커
                )
            )
        StockNameAll.objects.bulk_create(name_df_bulk,ignore_conflicts=True)
        print("종목명 갱신완료")

        for ticker in tqdm(tickers_all):

            # DB조회시 값이 있는지 확인할것
            #  있다면,
            if len(pd.DataFrame(StockPriceDateBase.objects.filter(ticker=ticker).values_list()))>0:
                print("최초입력이 아닙니다.")
                db = pd.DataFrame(StockPriceDateBase.objects.filter(ticker=ticker).values_list())
                # 해당 종목의 DB에서 최종 일자를 가져옴
                db_date = db.iloc[-1,1]
                # 영업일 날짜 DB리스트를 가져옴
                business_day_list = pd.DataFrame(BusinessDayDate.objects.all().values_list()).iloc[:,1].tolist()

                # 해당종목의 최종일자가 DB리스트에서 어디에 위치하는지 확인
                date_idx = business_day_list.index(db_date)
                # 목표로 하는 그 다음일자의 인덱스 위치를 선언
                target_date_idx = date_idx + 1
                # 만약 목표로 하는 인덱스가 리스트 범위를 벗어나는 경우의 예외처리
                if target_date_idx==len(business_day_list):
                    print("이미 최신 데이터입니다.")
                    continue
                start_date = business_day_list[target_date_idx]

            # 시작날짜가 없다면 초기값을 "19900101"로 설정
            else:
             start_date = "19000101"
             

            # 만약 지금 시간이 4시가 지났다면? 오늘날짜까지 갱신할것 아니라면 어제 날짜 데이터까지 가져올것
            if datetime.datetime.today().hour>=16:
                end_date = date_from_now(0)
            else:
                end_date = date_from_now(1)

            print(start_date)
            print(end_date)

            # 최신데이터라면 그냥 종료할것
            if start_date == end_date: 
                print("이미 최신의 데이터입니다. 저장을 종료합니다.")
            else: 
                # 정해진 날짜를 기반으로 조회 시작
                df_ohlcv = stock.get_market_ohlcv(start_date, end_date, ticker)
                stock_name = df_for_name[df_for_name.index==ticker]['종목명'].values[0]

                # df 재조정
                df_ohlcv['종목명'] = stock_name
                df_ohlcv['날짜'] = df_ohlcv.index
                df_ohlcv['등락률'] = df_ohlcv['등락률'].apply(lambda x : round(x,2))
                df_ohlcv['날짜'] = df_ohlcv['날짜'].apply(lambda x : str(x).replace("-","")[:8])
                df_ohlcv['티커'] = ticker
                # DB에 저장

                df_bulk =[]
                for i in tqdm(range(len(df_ohlcv))):
                    df_bulk.append(
                        StockPriceDateBase(
                        date             = df_ohlcv['날짜'].iloc[i],
                        name             = df_ohlcv['종목명'].iloc[i],   # 종목명
                        ticker           = df_ohlcv['티커'].iloc[i],  # 티커
                        start_price      = df_ohlcv['시가'].iloc[i],   # 시가
                        high_price       = df_ohlcv['고가'].iloc[i],   # 고가
                        low_price        = df_ohlcv['저가'].iloc[i],   # 저가
                        end_price        = df_ohlcv['종가'].iloc[i],   # 종가
                        fluctuation_rate = df_ohlcv['등락률'].iloc[i],   # 등락률
                        volume           = df_ohlcv['거래량'].iloc[i],   # 거래량
                        )
                    )
                StockPriceDateBase.objects.bulk_create(df_bulk,ignore_conflicts=True)

        print("===================저장완료=======================")
        return Response(status=200)
# 주가 패턴을 만듬
class CreatePattern(APIView):
    def get(self, request):
        # STEP 0 ===========================================
        # 상장한지 1년이 안된 신규상장주는 분석기능 미제공 
        # StockPricePattern.objects.all().delete()
        # 주가 데이터를 모두 가져옴

        StockPriceDB_df = StockPriceDateBase.objects.all()
        StockPriceDB_df = pd.DataFrame(StockPriceDB_df.values_list()).iloc[:,1:]
        StockPriceDB_df.columns = ['날짜','종목명','티커','시가','고가','저가','종가','거래량','등락률']

        # 종목명+ 티커 형태의 df를 불러옴
        # 주가 종목명 + 티커 형태의 df를 분리
        # 종목명 기준으로 반복하여 패턴을 생성할 것
        # df_name_list = StockPriceDB_df['종목명'].unique()[3290:]
        df_name_list = StockPriceDB_df['종목명'].unique()[12:]


        # 주가의 1년 패턴을 분해한다. [날짜, 종목명, 종가, 등락률]
        
        # 분석기간 설정
        analysis_period = 224

        for stock_name in tqdm(df_name_list):
            # STEP 1 ===========================================
            # 날짜순 정렬
            StockPriceDB_df_stocks = StockPriceDB_df[StockPriceDB_df['종목명']==stock_name]
            StockPriceDB_df_stocks = StockPriceDB_df_stocks.sort_values(by="날짜", ascending=True)

            # 종가데이터만을 받음
            StockPriceDB_df_stocks = StockPriceDB_df_stocks[['날짜', '종목명', '종가', '등락률','거래량']]
            # 전일 대비 거래량
            StockPriceDB_df_stocks['rate_volume'] = (StockPriceDB_df_stocks['거래량'].shift(periods=1)/StockPriceDB_df_stocks['거래량']-1).apply(lambda x: round(x,6))
            # 224일 평균선
            StockPriceDB_df_stocks['MA224'] = StockPriceDB_df_stocks['종가'].rolling(window=analysis_period).mean()
            # 주가 / 224일 ratio
            StockPriceDB_df_stocks['MA224_ratio'] = (StockPriceDB_df_stocks['종가']/StockPriceDB_df_stocks['MA224']).apply(lambda x: round(x,6))
            # 224일 표준편차(평균수익률)
            StockPriceDB_df_stocks['MA224_mean'] = StockPriceDB_df_stocks['등락률'].rolling(window=analysis_period).mean().apply(lambda x: round(x,6))
            # 224일 표준편차(위험)
            StockPriceDB_df_stocks['MA224_std'] = StockPriceDB_df_stocks['등락률'].rolling(window=analysis_period).std().apply(lambda x: round(x,6))
            # 5일후 수익률
            StockPriceDB_df_stocks['yeild_5days'] = (StockPriceDB_df_stocks['종가'].shift(periods=-5)/StockPriceDB_df_stocks['종가']-1).apply(lambda x: round(x,6))
            # 20일후 수익률
            StockPriceDB_df_stocks['yeild_20days'] = (StockPriceDB_df_stocks['종가'].shift(periods=-20)/StockPriceDB_df_stocks['종가']-1).apply(lambda x: round(x,6))
            # 60일후 수익률
            StockPriceDB_df_stocks['yeild_60days'] = (StockPriceDB_df_stocks['종가'].shift(periods=-60)/StockPriceDB_df_stocks['종가']-1).apply(lambda x: round(x,6))
            
            ## STEP 2 날짜기준으로 각각 pivoting =============================================
            

            # 주가기준
            StockPriceDB_df_stocks_price = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="종가")
            # 전일대비 거래량 기준
            StockPriceDB_df_stocks_volume_rate = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="rate_volume")
            # 등락률 기준
            StockPriceDB_df_stocks_rate = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="등락률")
            # MA224 기준
            StockPriceDB_df_stocks_MA224 = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224")
            # 주가/MA224 기준
            StockPriceDB_df_stocks_MA224_ratio = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224_ratio")
            # 224일 평균수익률 기준
            StockPriceDB_df_stocks_MA224_mean = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224_std")
            # 224일 평균위험 기준
            StockPriceDB_df_stocks_MA224_std = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="MA224_mean")
            # 5일후 수익률
            StockPriceDB_df_stocks_5days = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="yeild_5days")
            # 20일후 수익률
            StockPriceDB_df_stocks_20days = StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="yeild_20days")
            # 60일후 수익률
            StockPriceDB_df_stocks_60days =  StockPriceDB_df_stocks.pivot(index="종목명",columns="날짜",values="yeild_60days")



        #     # pivot DateFrame 초기화
            pivotted_StockPriceDB_df_stocks_data_result = pd.DataFrame()

        #     #  패턴 기간 설정  ================================================================================
            pattern_period = 244
        #     #  ==========================================================================================

        #     ## STEP 3 패턴 데이터 생성 =============================================
            print("패턴데이터 생성 시작")
            for idx in tqdm(range(len(StockPriceDB_df_stocks)-pattern_period)):
                # ------------------기초데이터 추출
                stock_name = StockPriceDB_df_stocks_price.iloc[:,idx:idx+pattern_period].index[0]
                # 컬럼 첫번째 날짜 
                start_day = StockPriceDB_df_stocks_price.iloc[:,idx:idx+pattern_period].columns[1]
                # 컬럼 마지막 날짜
                end_day = StockPriceDB_df_stocks_price.iloc[:,idx:idx+pattern_period].columns[-1]

                # ------------------ 데이터 제작
                # 주가/MA224  결과 데이터

                # 값이 NAN인경우에는 건너뛴다.
                temp_df_fluctuatuon_rate      = StockPriceDB_df_stocks_rate.iloc[:,idx:idx+pattern_period].copy()
                temp_df_volume_rate           = StockPriceDB_df_stocks_volume_rate.iloc[:,idx:idx+pattern_period].copy()
                temp_df_MA224                 = StockPriceDB_df_stocks_MA224.iloc[:,idx:idx+pattern_period].copy()
                temp_df_MA224_ratio           = StockPriceDB_df_stocks_MA224_ratio.iloc[:,idx:idx+pattern_period].copy()
                temp_df_MA224_mean            = StockPriceDB_df_stocks_MA224_mean.iloc[:,idx:idx+pattern_period].copy()
                temp_df_MA224_std             = StockPriceDB_df_stocks_MA224_std.iloc[:,idx:idx+pattern_period].copy()
                temp_df_yield_5days           = StockPriceDB_df_stocks_5days.iloc[:,idx:idx+pattern_period].copy().values.tolist()[0][-1]
                temp_df_yield_20days          = StockPriceDB_df_stocks_20days.iloc[:,idx:idx+pattern_period].copy().values.tolist()[0][-1]
                temp_df_yield_60days          = StockPriceDB_df_stocks_60days.iloc[:,idx:idx+pattern_period].copy().values.tolist()[0][-1]
                
                # 결측값이 있는지 확인한다.
                nan_value_check_ma224            = sum(temp_df_MA224.isnull().sum(axis=0))
                nan_value_check_volume_rate      = sum(temp_df_volume_rate.isnull().sum(axis=0))
                nan_value_check_fluctuatuon_rate =  sum(temp_df_fluctuatuon_rate.isnull().sum(axis=0))
                nan_value_check_rate             =  sum(temp_df_MA224_ratio.isnull().sum(axis=0))
                nan_value_check_mean             =  sum(temp_df_MA224_mean.isnull().sum(axis=0))
                nan_value_check_std              =  sum(temp_df_MA224_std.isnull().sum(axis=0))
                

                if nan_value_check_ma224 > 0:
                    # 있다면 다음 루프로
                    continue

                if nan_value_check_fluctuatuon_rate > 0:
                    # 있다면 다음 루프로
                    continue

                if nan_value_check_volume_rate > 0:
                    # 있다면 다음 루프로
                    continue

                if nan_value_check_rate > 0:
                    # 있다면 다음 루프로
                    continue
                if nan_value_check_mean > 0:
                    # 있다면 다음 루프로
                    continue
                if nan_value_check_std > 0:
                    # 있다면 다음 루프로
                    continue
                # Dataframe을 List로 변경
                temp_df_fluctuatuon_rate     = temp_df_fluctuatuon_rate.copy().values.tolist()
                temp_df_MA224                = temp_df_MA224.copy().values.tolist()
                temp_df_volume_rate          = temp_df_volume_rate.copy().values.tolist()
                temp_df_MA224_ratio          = temp_df_MA224_ratio.copy().values.tolist()
                temp_df_MA224_mean           = temp_df_MA224_mean.copy().values.tolist()
                temp_df_MA224_std            = temp_df_MA224_std.copy().values.tolist()

                # df 초기화  
                temp_data_df = pd.DataFrame()

                # 주가 결과 데이터
                temp_df_price = StockPriceDB_df_stocks_price.iloc[:,idx:idx+pattern_period].copy().values.tolist()

            
                temp_data_df.loc[0,'종목명'] = stock_name
                temp_data_df.loc[0,'분석기간'] = pattern_period
                temp_data_df.loc[0,'시작일자'] = start_day
                temp_data_df.loc[0,'종료일자'] = end_day
                temp_data_df['주가_list'] = temp_df_price
                temp_data_df['등락률_list'] = temp_df_fluctuatuon_rate
                temp_data_df['전일대비_거래량_list'] = temp_df_volume_rate    
                temp_data_df['MA224_list'] = temp_df_MA224
                temp_data_df['주가/MA224_list'] = temp_df_MA224_ratio
                temp_data_df['rate_MA224_mean_list'] = temp_df_MA224_mean
                temp_data_df['rate_MA224_std_list'] = temp_df_MA224_std
                # 단기
                temp_data_df['yield_5days'] = temp_df_yield_5days
                # 중기
                temp_data_df['yield_20days'] = temp_df_yield_20days
                # 중장기
                temp_data_df['yield_60days'] = temp_df_yield_60days
                # ------------------데이터프레임에 추가
                pivotted_StockPriceDB_df_stocks_data_result = pd.concat([pivotted_StockPriceDB_df_stocks_data_result.copy(),temp_data_df],axis=0)

            # 역순정렬
            pivotted_StockPriceDB_df_stocks_data_result = pivotted_StockPriceDB_df_stocks_data_result.sort_values(by="종료일자",ascending=False)
            
            print("주가 데이터 수"+str(len(StockPriceDB_df_stocks)))
            print("패턴 데이터 수"+str(len(pivotted_StockPriceDB_df_stocks_data_result)))


            # DB에 저장된 패턴수 확인
            db_pattern_cnt = StockPricePattern.objects.filter(name=stock_name).count()
            # 현재 생성된 패턴수 확인
            current_pattern_cnt =len(pivotted_StockPriceDB_df_stocks_data_result)
            # 저장할 패턴의 수 설정
            saving_cnt = current_pattern_cnt-db_pattern_cnt

            if saving_cnt == 0 :
                print("이미 최신의 데이터입니다.")
                continue
            else:
                df_bulk =[]

                print("DB저장 시작")
                for i in tqdm(range(saving_cnt)):
                    # 중복되지 않은 경우에만 df_bulk에 넣을것
                    df_bulk.append(
                        StockPricePattern(
                    name             = pivotted_StockPriceDB_df_stocks_data_result['종목명'].iloc[i],  # 종목명
                    analysis_period  = pivotted_StockPriceDB_df_stocks_data_result['분석기간'].iloc[i],  # 분석기간
                    start_day        = pivotted_StockPriceDB_df_stocks_data_result['시작일자'].iloc[i],  # 시작일자
                    end_day          = pivotted_StockPriceDB_df_stocks_data_result['종료일자'].iloc[i],  # 종료일자
                    price_list       = pivotted_StockPriceDB_df_stocks_data_result['주가_list'].iloc[i],  # 주가_list
                    fluctuation_list = pivotted_StockPriceDB_df_stocks_data_result['등락률_list'].iloc[i],  # 등락률_list
                    rate_volume_list = pivotted_StockPriceDB_df_stocks_data_result['전일대비_거래량_list'].iloc[i],  # 등락률_list
                    MA224_list       = pivotted_StockPriceDB_df_stocks_data_result['MA224_list'].iloc[i],  # MA224_list
                    MA224_ratio_list  = pivotted_StockPriceDB_df_stocks_data_result['주가/MA224_list'].iloc[i],  # 주가/MA224_list
                    rate_MA224_mean_list  = pivotted_StockPriceDB_df_stocks_data_result['rate_MA224_mean_list'].iloc[i],  # rate_MA224_mean_list
                    rate_MA224_std_list   = pivotted_StockPriceDB_df_stocks_data_result['rate_MA224_std_list'].iloc[i],  # rate_MA224_std_list
                    yield_5days   = pivotted_StockPriceDB_df_stocks_data_result['yield_5days'].iloc[i],  # 5일후 수익률
                    yield_20days   = pivotted_StockPriceDB_df_stocks_data_result['yield_20days'].iloc[i],  # 20일후 수익률
                    yield_60days   = pivotted_StockPriceDB_df_stocks_data_result['yield_60days'].iloc[i],  # 60일후 수익률
                    ))

                StockPricePattern.objects.bulk_create(df_bulk,ignore_conflicts=True)
                print("저장완료")

        return Response(status=200)

class CreateCosim(APIView):
    def get(self, request):
        # 필요한 변수로는 분석종목명,분석기간(시작 및 종료일자를 뽑을것),
        # ==================입력 데이터 구간
        # 기본적으로 오늘 날짜를 가져옴
        end_date = date_from_now(0)
        stock_name = "AJ네트웍스"
        analysis_period = 244
        # 분석에 과거 몇년의 데이터를 사용할 것인지
        # analysis_year = 
        # 어떤 패턴으로 분석할 것인지 
        # analysis_type = 

        # ==================입력 데이터 구간



        # DB에서 영업일 조회
        date_df = pd.DataFrame(BusinessDayDate.objects.all().values_list())
        date_df.columns = ['id','date']
        date_df = date_df.sort_values(by='date',ascending=True)
        business_date_list = date_df.iloc[:,1].tolist()
        start_date = business_date_list[business_date_list.index(end_date)-487:business_date_list.index(end_date)][0]



        # 조회할 주가의 데이터를 뽑음
        stock_name_ticker_df = pd.DataFrame(StockNameAll.objects.filter(name=stock_name).values_list())
        stock_name_ticker_df.columns = ['id','name','ticker']
        ticker = stock_name_ticker_df['ticker'].values[0]

        # 조회한 종목 데이터 가공===================================
        stock_df = stock.get_market_ohlcv(start_date, end_date, ticker).reset_index()
        stock_df = stock_df[['날짜','종가','거래량','등락률']]
        stock_df['종목명'] = stock_name
        
        # 전일 대비 거래량
        stock_df['rate_volume'] = (stock_df['거래량'].shift(periods=1)/stock_df['거래량']-1).apply(lambda x: round(x,6))
        # 224일 평균선
        stock_df['MA224'] = stock_df['종가'].rolling(window=analysis_period).mean()
        # 주가 / 224일 ratio
        stock_df['MA224_ratio'] = stock_df['종가']/stock_df['MA224'].apply(lambda x: round(x,6))
        # 224일 표준편차(평균수익률)
        stock_df['MA224_mean'] = stock_df['등락률'].rolling(window=analysis_period).mean().apply(lambda x: round(x,6))
        # 224일 표준편차(위험)
        stock_df['MA224_std'] = stock_df['등락률'].rolling(window=analysis_period).std().apply(lambda x: round(x,6))

        stock_df = stock_df.dropna().iloc[1:]

        # 해당종목의 패턴을 생성
        # 주가기준
        StockPriceDB_df_stocks_price = stock_df.pivot(index="종목명",columns="날짜",values="종가").values.tolist()[0]
        # 전일대비 거래량 기준
        StockPriceDB_df_stocks_volume_rate = stock_df.pivot(index="종목명",columns="날짜",values="rate_volume").values.tolist()[0]
        # 등락률 기준
        StockPriceDB_df_stocks_rate = stock_df.pivot(index="종목명",columns="날짜",values="등락률").values.tolist()[0]
        # MA224 기준
        StockPriceDB_df_stocks_MA224 = stock_df.pivot(index="종목명",columns="날짜",values="MA224").values.tolist()[0]
        # 주가/MA224 기준
        StockPriceDB_df_stocks_MA224_ratio = stock_df.pivot(index="종목명",columns="날짜",values="MA224_ratio").values.tolist()[0]
        # 224일 평균수익률 기준
        StockPriceDB_df_stocks_MA224_mean = stock_df.pivot(index="종목명",columns="날짜",values="MA224_std").values.tolist()[0]
        # 224일 평균위험 기준
        StockPriceDB_df_stocks_MA224_std = stock_df.pivot(index="종목명",columns="날짜",values="MA224_mean").values.tolist()[0]

        print("조회 종목 패턴 생성 완료")
        
        db_pattern_df = pd.DataFrame(StockPricePattern.objects.all().values_list())
        # db_pattern_df = pd.DataFrame(StockPricePattern.objects.filter(start_day__startswith="2022",).values_list())

        # 구분을 columns 설정
        db_pattern_df.columns = ['id','종목명','분석일자','시작일자','종료일자','주가_list','등락률_list','전일대비_거래량_list','MA224_list','주가/MA224_list','rate_MA224_mean_list','rate_MA224_std_list','yield_5days','yield_20days','yield_60days']


        # # 거래량 패턴만 가지고 분석해보기
        # db_pattern_df_volume_rate = db_pattern_df[['종목명','시작일자','종료일자','전일대비_거래량_list','등락률_list','yield_5days','yield_20days','yield_60days']]
        # db_pattern_df_volume_rate['cosim_rate'] = db_pattern_df_volume_rate['전일대비_거래량_list'].apply(lambda x : cos_sim(np.array(literal_eval(str(StockPriceDB_df_stocks_volume_rate))),np.array(literal_eval(str(x)))))
        # db_pattern_df_volume_rate = db_pattern_df_volume_rate.sort_values(by="cosim_rate", ascending=False)
        # print("전일대비 거래량 패턴 기준")
        # print(db_pattern_df_fluctuation_rate.head(10))

        # 등락률 패턴만 가지고 분석해보기
        db_pattern_df_fluctuation_rate = db_pattern_df[['종목명','시작일자','종료일자','주가_list','등락률_list','yield_5days','yield_20days','yield_60days']]
        db_pattern_df_fluctuation_rate['cosim_rate'] = db_pattern_df_fluctuation_rate['등락률_list'].apply(lambda x : cos_sim(np.array(literal_eval(str(StockPriceDB_df_stocks_rate))),np.array(literal_eval(str(x)))))
        db_pattern_df_fluctuation_rate = db_pattern_df_fluctuation_rate.sort_values(by="cosim_rate", ascending=False)
        db_pattern_df_fluctuation_rate.drop_duplicates(subset='종목명',keep='first',inplace=True)
        print("등락률 패턴 기준")
        print(db_pattern_df_fluctuation_rate.head(10))

        # 주가/MA224_list 패턴만 가지고 분석해보기
        db_pattern_df_MA224_ratio = db_pattern_df[['종목명','시작일자','종료일자','주가_list','주가/MA224_list','yield_5days','yield_20days','yield_60days']]
        db_pattern_df_MA224_ratio['cosim_rate'] = db_pattern_df_MA224_ratio['주가/MA224_list'].apply(lambda x : cos_sim(np.array(literal_eval(str(StockPriceDB_df_stocks_MA224_ratio))),np.array(literal_eval(str(x)))))
        db_pattern_df_MA224_ratio = db_pattern_df_MA224_ratio.sort_values(by="cosim_rate", ascending=False)
        db_pattern_df_MA224_ratio.drop_duplicates(subset='종목명',keep='first',inplace=True)
        print("주가/MA224_비율 패턴 기준")
        print(db_pattern_df_MA224_ratio.head(10))
        
        return Response(status=200)


# 유사도가 높은 패턴을 5개씩 뽑고, 진단(단,중장기 예측)
# 224일 rate 유사도, 변동률 유사도, 전일비 거래량 유사도의 가중치 조정을 할것, 투자판단까지 가능하도록 하여, 투자수익률을 구할수 있도록 함


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





# # 종목 관련 함수
# def get_stock_basic_info(day=0, market="ALL", detail="ALL"):
#     """ 종목 기초정보 제공 함수
#         티커/종목명/시가/종가/변동폭(오늘)/등락률(오늘)/\n
#         거래량(오늘)/거래대금(오늘)/상장주식수/보유수량/\n
#         지분율/한도수량/한도소진률/BPS/PER/PBR/EPS/DIV/DPS_
        
#         All = 모든 정보
#         BASIC = 기초 정보

#     Args:
#         day (int, optional): _description_. Defaults to 0.
#         market (str, optional): _description_. Defaults to "ALL".
#         detail (str, optional): _description_. Defaults to "ALL".

#     Returns:
#         _type_: DataFrame
#     """
#     if detail=="ALL":
#         # day일전(영업일기준) 일자를 불러옴
#         df_name = stock.get_market_price_change(date_from_now(day),date_from_now(day),market="ALL").reset_index()[['종목명','티커']]
#         df_basic = stock.get_market_ohlcv(date_from_now(day), market=market).reset_index()
#         df_fundamental = stock.get_market_fundamental( date_from_now(day), market=market).reset_index()
#         df_result = pd.merge(df_basic,df_fundamental, on='티커',how='left')
#         df_result = pd.merge(df_result,df_name, on='티커',how='left')
#         # 일자의 문자화
#         str_date = str(date_from_now(day))
#         df_result.loc[:, '일자'] = str_date[0:4]+"-"+str_date[4:6]+"-"+str_date[6:]
#         df_result['티커'] = df_result['티커'].astype(dtype='object')
#         df_result =df_result[['일자','종목명','티커','시가','고가','저가','종가','등락률','거래량','거래대금','PER','BPS','PBR','EPS','DIV','DPS']]     
#         df_result = df_result.where(pd.notnull(df_result), None)
#         return df_result
#     if detail=="BASIC":
#         df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
#         return df_change

# class DBcheck(APIView):
#     def get(self, request):
#         # DB연결 확인
#         db_connection_path = 'mysql+pymysql://y2kwlswn:wjd7615@localhost:8000/StockPriceDB'
#         db_connection = create_engine(db_connection_path)

#         conn = db_connection.connect()
#         return Response(status=200)