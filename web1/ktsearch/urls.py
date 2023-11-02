from django.conf.urls import include, re_path

from . import views

urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/search/$',
            views.SearchView.as_view(),
            name='kts_search',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/summary/(?P<kt_search_id>[0-9]+)/$',
            views.SummaryView.as_view(),
            name='kts_summary',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/resolve/(?P<kt_search_id>[0-9]+)/$',
            views.ResolveView.as_view(),
            name='kts_resolve',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/name_resolve/(?P<kt_search_id>[0-9]+)/$',
            views.NameResolveView.as_view(),
            name='kts_name_resolve',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/bulk_name_resolve/(?P<kt_search_id>[0-9]+)/$',
            views.BulkNameResolveView.as_view(),
            name='kts_bulk_name_resolve',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/faers_data/$',
            views.SearchViewFaersTable.as_view(),
            name='kts_search_faers_data',
            ),
    ]
    
