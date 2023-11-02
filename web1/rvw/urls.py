from django.conf.urls import re_path

from . import views

app_name = 'rvw'
urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/review/$',
            views.ReviewView.as_view(), name='review'),
    re_path(r'^(?P<ws_id>[0-9]+)/review_summary/$',
            views.ReviewSummaryView.as_view(), name='review_summary'),
    re_path(r'^(?P<ws_id>[0-9]+)/election/(?P<elec_id>[0-9]+)/$',
            views.ElectionView.as_view(), name='election'),
    re_path(r'^(?P<ws_id>[0-9]+)/election/summary/(?P<wsa_ids>[0-9,]+)/$',
            views.get_drug_summary, name='get_drug_summary'),
    re_path(r'^all_review_notes/$',
            views.AllReviewNotesView.as_view(), name='all_review_notes'),
    re_path(r'^(?P<ws_id>[0-9]+)/prescreen/(?P<wsa_id>[0-9]+)/$',
            views.Prescreen2View.as_view(), name='prescreen'),
    re_path(r'^(?P<ws_id>[0-9]+)/hitclusters/$',
            views.HitClusterView.as_view(), name='hitclusters'),
    re_path(r'^(?P<ws_id>[0-9]+)/animal_model_compare/$',
            views.AnimalModelCompareView.as_view(), name='animal_model_compare'),
    re_path(r'^(?P<ws_id>[0-9]+)/defus_details/$',
            views.DefusDetailsView.as_view(), name='defus_details'),
    ]

