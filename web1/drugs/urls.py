from django.conf.urls import include, re_path

from . import views

urlpatterns = [
    re_path(r'^edit/$',
            views.DrugEditView.as_view(),
            name='drug_edit_view',
            ),
    re_path(r'^review/$',
            views.DrugEditReviewView.as_view(),
            name='drug_edit_review_view',
            ),
    re_path(r'^proposal/(?P<proposal_id>[0-9]+)/$',
            views.proposal_data,
            name='drug_proposal_data',
            ),
    re_path(r'^resolve/(?P<proposal_id>[0-9]+)/(?P<resolution>[0-9]+)/$',
            views.resolve_proposal,
            name='resolve_proposal',
            ),
    re_path(r'^changes/$',
            views.DrugChangesView.as_view(),
            name='drug_changes_view',
            ),
    re_path(r'^search/$',
            views.DrugSearchView.as_view(),
            name='drug_search_view',
            ),
    re_path(r'^twoxar_attrs/$', views.twoxar_attrs,),
    re_path(r'^twoxar_dpi/$', views.twoxar_dpi,),
    re_path(r'^chem_image/(?P<drug_id>[0-9]+)/',
            views.mol_chem_image, name='chem_image'),
    re_path(r'^import_molecule/$', views.import_molecule,),
]
