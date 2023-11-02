from django.conf.urls import re_path

from . import views

# XXX Eventually, all views labeled ws_admin, ds_ps, and qc in the output from
# XXX experiments/view_dir could move here.

app_name = 'wsmgr'
urlpatterns = [
    re_path(r'create_ws/',
            views.CreateWsView.as_view(), name='create_ws'),
    re_path(r'^(?P<ws_id>[0-9]+)/copy_ws/',
            views.CopyWsView.as_view(), name='copy_ws'),
    re_path(r'^(?P<ws_id>[0-9]+)/imphist/',
            views.ImportHistoryView.as_view(), name='imphist'),
    ]

