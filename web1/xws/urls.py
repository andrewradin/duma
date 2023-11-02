
from django.conf.urls import include, re_path

from . import views

app_name='xws'
urlpatterns = [
    re_path(r'^ongoct/$', views.OngoCTView.as_view(), name='ongoct'),
    re_path(r'^retroct/$', views.RetroCTView.as_view(), name='retroct'),
]
