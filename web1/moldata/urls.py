from django.conf.urls import re_path

from . import views,imp_views

app_name = 'moldata'
urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/annotate/(?P<wsa_id>[0-9]+)/',
            views.AnnotateView.as_view(), name='annotate'),
    re_path(r'^(?P<ws_id>[0-9]+)/drug_run_detail/(?P<wsa_id>[0-9]+)/',
            views.DrugRunDetailView.as_view(), name='drug_run_detail'),
    re_path(r'^(?P<ws_id>[0-9]+)/drugcmp/',
            views.DrugCmpView.as_view(), name='drugcmp'),
    re_path(r'^(?P<ws_id>[0-9]+)/molcmp/',
            views.MolCmpView.as_view(), name='molcmp'),
    re_path(r'^(?P<ws_id>[0-9]+)/assays/(?P<wsa_id>[0-9]+)/',
            views.AssaysView.as_view(), name='assays'),
    re_path(r'^(?P<ws_id>[0-9]+)/noneff_assays/(?P<wsa_id>[0-9]+)/',
            views.NoneffAssaysView.as_view(), name='noneff_assays'),
    re_path(r'^(?P<ws_id>[0-9]+)/patent_detail/(?P<wsa_id>[0-9]+)/',
            views.PatentDetailView.as_view(), name='patent_detail'),
    re_path(r'^(?P<ws_id>[0-9]+)/dispositionaudit/(?P<wsa_id>[0-9]+)/',
            views.DispositionAuditView.as_view(), name='dispositionaudit'),
    re_path(r'^(?P<ws_id>[0-9]+)/trgimp/(?P<wsa_id>[0-9]+)/',
            imp_views.TrgImpView.as_view(), name='trgimp'),
    re_path(r'^(?P<ws_id>[0-9]+)/scrimp/(?P<wsa_id>[0-9]+)/',
            imp_views.ScrImpView.as_view(), name='scrimp'),
    re_path(r'^(?P<ws_id>[0-9]+)/trg_scr_imp/(?P<wsa_id>[0-9]+)/',
            imp_views.TrgScrImpView.as_view(), name='trg_scr_imp'),
    re_path(r'^(?P<ws_id>[0-9]+)/ind_trg_imp/(?P<wsa_id>[0-9]+)/',
            imp_views.IndTrgImpView.as_view(), name='ind_trg_imp'),
    re_path(r'^(?P<ws_id>[0-9]+)/hit_selection/',
            views.HitSelectionView.as_view(), name='hit_selection'),
    re_path(r'^(?P<ws_id>[0-9]+)/hit_selection_report/',
            views.HitSelectionReportView.as_view(), name='hit_selection_report'),
    ]

