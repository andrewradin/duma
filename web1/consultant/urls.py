from django.conf.urls import include, re_path

from . import views

urlpatterns = [
    re_path(r'^$',
            views.ConsultantView.as_view(),
            name='consultant_view',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/molecule/(?P<wsa_id>[0-9]+)/$',
            views.ConsultantMoleculeView.as_view(),
            name='consultant_molecule',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/protein/(?P<prot_id>[A-Z0-9-]+)/$',
            views.ConsultantProteinView.as_view(),
            name='consultant_protein',
            ),
]
