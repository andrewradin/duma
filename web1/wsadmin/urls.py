from django.conf.urls import re_path

from . import views

app_name = 'wsadmin'
urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/custom_dpi/',
            views.CustomDpiView.as_view(), name='custom_dpi'),
    ]

