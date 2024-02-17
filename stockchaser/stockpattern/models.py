from django.db import models

# Create your models here.
class StockPriceDateBase(models.Model):
    date             = models.TextField()   # 날짜 
    name             = models.TextField()   # 종목명
    ticker           = models.TextField()  # 티커
    start_price      = models.IntegerField()   # 시가
    high_price       = models.IntegerField()   # 고가
    low_price        = models.IntegerField()   # 저가
    end_price        = models.IntegerField()   # 종가
    fluctuation_rate = models.FloatField()   # 등락률
    volume           = models.IntegerField()   # 거래량
    volume_amount    = models.IntegerField()   # 거래대금
    # -값이 나올수 있는 per 때문에 지표들은 모두 text 처리를 우선으로 함
    per              = models.TextField()   # per
    bps              = models.TextField()   # bps
    pbr              = models.TextField()   # pbr
    eps              = models.TextField()   # eps
    div              = models.TextField()   # div
    dps              = models.TextField()   # dps