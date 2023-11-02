from django.conf.urls import re_path

from . import views
from . import search_views as sv
from . import ts_views as tsv
from . import qual_views as qv

app_name = 'ge'
urlpatterns = [
    re_path(r'^([0-9]+)/tissues/', views.tissues, name='tissues'),
    re_path(r'^(?P<ws_id>[0-9]+)/note_tissue/(?P<tissue_id>[0-9]+)/',
            views.NoteTissueView.as_view(), name='note_tissue'),
    re_path(r'^(?P<ws_id>[0-9]+)/classify/(?P<tissue_id>[0-9]+)/$',
            views.ClassifyView.as_view(), name='classify'),
    re_path(r'^(?P<ws_id>[0-9]+)/ae_search/',
            sv.AeSearchView.as_view(), name='ae_search'),
    re_path(r'^([0-9]+)/ae_list/([0-9]+)/', sv.ae_list, name='ae_list'),
    re_path(r'^(?P<ws_id>[0-9]+)/ae_bulk/(?P<ae_search_id>[0-9]+)/',
            sv.AeBulkView.as_view(), name='ae_bulk'),
    re_path(r'^(?P<ws_id>[0-9]+)/tissue_set/(?P<ts_id>[0-9]+)/',
            tsv.tissue_set, name='tissue_set'),
    re_path(r'^([0-9]+)/tissue_set/',
            tsv.tissue_set, name='tissue_set_create'),
    re_path(r'^(?P<ws_id>[0-9]+)/tissue_corr/',
            tsv.TissueCorrView.as_view(), name='tissue_corr'),
    re_path(r'^(?P<ws_id>[0-9]+)/tissue_set_analysis/',
            tsv.TissueSetAnalysisView.as_view(), name='tissue_set_analysis'),
    re_path(r'^(?P<ws_id>[0-9]+)/kt_tiss/(?P<ts_id>[0-9]+)/$',
            tsv.kt_tiss, name='kt_tiss'),
    re_path(r'^(?P<ws_id>[0-9]+)/tissue_stats/(?P<ts_id>[0-9]+)/$',
            tsv.tissue_stats, name='tissue_stats'),
    re_path(r'^(?P<ws_id>[0-9]+)/sigprot/(?P<tissue_id>[0-9]+)/',
            qv.SigProtView.as_view(), name='sigprot'),
    re_path(r'^(?P<ws_id>[0-9]+)/sig_qc/(?P<tissue_id>[0-9]+)/$',
            qv.sigQcView.as_view(), name='sig_qc'),
    re_path(r'^(?P<ws_id>[0-9]+)/ge_overlap/$', qv.ge_overlap, name='ge_overlap'),
    re_path(r'^(?P<ws_id>[0-9]+)/search_model/(?P<job_id>[0-9]+)/$' , sv.SearchModelView.as_view(), name='searchmodel'),
    ]

