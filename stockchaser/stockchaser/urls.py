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
from stockpattern.loader import CreatePattern,LoadRecentData, CreateCosim

urlpatterns = [
    # 홈페이지 url
    path('', MainPage.as_view()),

    # 기능 url

    ### 데이터 로드
    path('loaddata/', LoadRecentData.as_view()),
    # 패턴 만들기
    path('createpattern',CreatePattern.as_view()), 

    path('cosim',CreateCosim.as_view()),  

    # admin 관리
    path('admin/', admin.site.urls),
]
