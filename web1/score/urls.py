
from django.conf.urls import include, re_path

from . import views

app_name='score'
urlpatterns = [
    re_path(r'^(?P<ws_id>[0-9]+)/score_synergy/',
                    views.ScoreSynergyView.as_view(),
                    name='score_synergy',
                    ),
    re_path(r'^(?P<ws_id>[0-9]+)/feat_pair_heat/',
                    views.FeaturePairHeatView.as_view(),
                    name='feat_pair_heat',
                    ),
    re_path(r'^(?P<ws_id>[0-9]+)/feat_pair/',
                    views.FeaturePairView.as_view(),
                    name='feat_pair',
                    ),
    re_path(r'^(?P<ws_id>[0-9]+)/mrmr_cmp/',
                    views.MRMRCmpView.as_view(),
                    name='mrmr_cmp',
                    ),
    re_path(r'^(?P<ws_id>[0-9]+)/weight_cmp/',
                    views.WeightCmpView.as_view(),
                    name='weight_cmp',
                    ),
]
