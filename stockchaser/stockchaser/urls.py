"""
URL configuration for stockchaser project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

from stockpattern.views import MainPage
from stockpattern.loader import LoadPrice, LoadRecentPrice, DeletePrice#, DBcheck

urlpatterns = [
    # 홈페이지 url
    path('', MainPage.as_view()),

    # 기능 url
    ### 데이터 로드
    path('load',LoadPrice.as_view()),    
    ### 최근 데이터 로드
    path('recentload',LoadRecentPrice.as_view()),    

    ### 데이터 삭제
    path('delete',DeletePrice.as_view()),

    # admin 관리
    path('admin/', admin.site.urls),
]
