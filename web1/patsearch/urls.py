from django.conf.urls import include, re_path

from . import views

urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/search/$',
            views.SearchView.as_view(),
            name='pats_search',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/summary/(?P<pat_search_id>[0-9]+)/$',
            views.SummaryView.as_view(),
            name='pats_summary',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/resolve/(?P<pat_dd_search_id>[0-9]+)/$',
            views.ResolveView.as_view(),
            name='pats_resolve',
            ),

    re_path(r'^(?P<ws_id>[0-9]+)/preview_search/$',
            views.preview_search,
            name='preview_search',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/patent_details/(?P<patent_result_id>[0-9]+)/$',
            views.patent_details,
            name='patent_details',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/resolve/(?P<patent_result_id>[0-9]+)/(?P<resolution>[0-9]+)/$',
            views.resolve_patent,
            name='resolve_patent',
            ),

    re_path(r'^(?P<ws_id>[0-9]+)/drugset/(?P<setname>[0-9a-zA-Z_ ]+)/$',
            views.get_drugset,
            name='drugset_terms',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/drug/(?P<wsa_id>[0-9]+)/$',
            views.get_drug,
            name='drug_terms',
            ),

    re_path(r'^(?P<ws_id>[0-9]+)/search_drugs/(?P<query>[0-9a-zA-Z_]+)/$',
            views.search_drugs,
            name='search_drugs',
            ),
    ]
    
