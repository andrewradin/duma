from django.conf.urls import include, re_path

from . import views

urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/trial_drugs/$',
            views.TrialDrugsView.as_view(),
            name='cts_trial_drugs',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/ct_resolve/(?P<search_id>[0-9]+)/$',
            views.CtResolveView.as_view(),
            name='cts_resolve',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/ct_summary/(?P<search_id>[0-9]+)/$',
            views.CtSummaryView.as_view(),
            name='cts_summary',
            ),
    ]
    
