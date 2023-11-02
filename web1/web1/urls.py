from django.conf.urls import include, re_path

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

import browse.api as ba
import browse.views as bv
import runner.views as rv
import notes.views as nv
import flagging.views as fv
import moldata.views as mv
import django.contrib.auth.views as av

from two_factor.urls import urlpatterns as tf_urls

urlpatterns = [
    re_path(r'', include(tf_urls, namespace="two_factor")),
    re_path(r'^admin/', admin.site.urls),
    # delegations to duma sub-apps; everything listed here should also
    # be listed in the prefix_list in selenium/duma.py
    re_path(r'^cv/', include('nav.urls')),
    re_path(r'^cts/', include('ctsearch.urls')),
    re_path(r'^kts/', include('ktsearch.urls')),
    re_path(r'^pats/', include('patsearch.urls')),
    re_path(r'^drugs/', include('drugs.urls')),
    re_path(r'^consultant/', include('consultant.urls')),
    re_path(r'^mol/', include('moldata.urls')),
    re_path(r'^ge/', include('ge.urls')),
    re_path(r'^rvw/', include('rvw.urls')),
    re_path(r'^wsadmin/', include('wsadmin.urls')),
    re_path(r'^wsmgr/', include('wsmgr.urls')),
    re_path(r'^score/', include('score.urls')),
    re_path(r'^xws/', include('xws.urls')),
    re_path(r'^(?P<ws_id>[0-9]+)/annotate/(?P<wsa_id>[0-9]+)/',
            mv.AnnotateView.as_view(), name='annotate_redirect'),
    re_path(r'^jobs/', rv.jobs),
    re_path(r'^jobsum/', bv.jobsum,name='nws_jobsum'),
    re_path(r'^job_detail/', bv.job_detail),
    re_path(r'^users/', bv.users,name='nws_users'),
    re_path(r'^mem/', bv.MemView.as_view()),
    re_path(r'^s3_cache/', bv.S3CacheView.as_view(), name='s3_cache'),
    re_path(r'^coll_stats/', bv.CollStatsView.as_view(), name='coll_stats'),
    re_path(r'^etl_status/', bv.EtlStatusView.as_view(), name='etl_status'),
    re_path(r'^etl_order/', bv.EtlOrderView.as_view(), name='etl_order'),
    re_path(r'^etl_history/(?P<etl_dir>.+)/',
            bv.EtlHistoryView.as_view(), name='etl_history'),
    re_path(r'^credits/', bv.CreditsView.as_view(), name='credits'),
    re_path(r'^(?P<ws_id>[0-9]+)/users/', bv.users, name='users'),
    re_path(r'^note/(?P<note_id>[0-9]+)', nv.history, name='note_hist'),
    re_path(r'^upload/'
            , bv.UploadView.as_view(), name='upload'),
    re_path(r'^(?P<ws_id>[0-9]+)/workflow/', bv.WorkflowView.as_view(), name='workflow'),
    re_path(r'^(?P<ws_id>[0-9]+)/data_status/$' , bv.DataStatusView.as_view(), name='data_status'),
    re_path(r'^(?P<ws_id>[0-9]+)/patent_notes/$' , bv.PatentNotesView.as_view(), name='patent_notes'),
    re_path(r'^(?P<ws_id>[0-9]+)/review_notes/$' , bv.ReviewNotesView.as_view(), name='review_notes'),
    re_path(r'^(?P<ws_id>[0-9]+)/comp_evidence/$' , bv.CompEvidenceView.as_view(), name='comp_evidence_view'),
    re_path(r'^(?P<ws_id>[0-9]+)/competition/$' , bv.CompetitionView.as_view(), name='competition'),
    #re_path(r'^(?P<ws_id>[0-9]+)/test/$' , bv.test, name='test'),
    re_path(r'^(?P<ws_id>[0-9]+)/test/$' , bv.DumaTestView.as_view(), name='test'),
    #re_path(r'^(?P<ws_id>[0-9]+)/test_form/$' , bv.FormTestView.as_view(), name='test_form'),
    re_path(r'^(?P<ws_id>[0-9]+)/test_form/$' , bv.DumaFormTestView.as_view(), name='test_form'),
    re_path(r'^(?P<ws_id>[0-9]+)/faers/(?P<job_id>[0-9]+)/$' , bv.FaersView.as_view(), name='faers'),
    re_path(r'^(?P<ws_id>[0-9]+)/faers/$' , bv.FaersView.as_view(), name='faers_base'),
    re_path(r'^(?P<ws_id>[0-9]+)/faers_indi/$' , bv.FaersIndiView.as_view(), name='faers_indi'),
    re_path(r'^(?P<ws_id>[0-9]+)/faers_run_table/$' , bv.FaersRunTable.as_view(), name='faers_run_table'),
    re_path(r'^(?P<ws_id>[0-9]+)/faers_demo_view/$' , bv.FaersDemoView.as_view(), name='faers_demo'),
    re_path(r'^(?P<ws_id>[0-9]+)/plotly/$' , bv.plotly, name='plotly'),
    re_path(r'^(?P<ws_id>[0-9]+)/pca/$' , bv.PcaView.as_view(), name='pca'),
    re_path(r'^(?P<ws_id>[0-9]+)/gwas_search/$' , bv.GwasSearchView.as_view(), name='gwas_search'),
    re_path(r'^(?P<ws_id>[0-9]+)/gwas_qc/$' , bv.GwasQcView.as_view(), name='gwas_qc'),
    re_path(r'^(?P<ws_id>[0-9]+)/flagset/$' , fv.FlagSetView.as_view(), name='flagset'),
    re_path(r'^(?P<ws_id>[0-9]+)/drugset/$', bv.DrugSetView.as_view(), name='drugset'),
    re_path(r'^(?P<ws_id>[0-9]+)/clust_screen/',
            bv.ClustScreenView.as_view(),
            name='clust_screen',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/pathways/', bv.PathwaysView.as_view(), name='pathways'),
    re_path(r'^(?P<ws_id>[0-9]+)/proteinscores/', bv.ProteinScoreView.as_view(), name='protein_scores'),
    re_path(r'^(?P<ws_id>[0-9]+)/proteinnetwork/', bv.ProteinNetworkView.as_view(), name='protein_network'),
    re_path(r'^pathway_network/', bv.PathwayNetworkView.as_view(), name='pathway_network'),
    re_path(r'^(?P<ws_id>[0-9]+)/protein/(?P<prot_id>[A-Z0-9-]+)/',
            bv.ProteinView.as_view(),
            name='protein',
            ),
    re_path(r'^(?P<ws_id>[0-9]+)/prot_detail/(?P<prot_id>[A-Z0-9-]+)/'
            , bv.ProtDetailView.as_view(), name='prot_detail'),
    re_path(r'^(?P<ws_id>[0-9]+)/prot_search/$'
            , bv.prot_search, name='prot_search'),
    re_path(r'^(?P<ws_id>[0-9]+)/target_data/'
            , bv.TargetDataView.as_view(), name='target_data'),
    re_path(r'^(?P<ws_id>[0-9]+)/wf_diag/'
            , bv.WfDiagView.as_view(), name='wf_diag'),
    re_path(r'^(?P<ws_id>[0-9]+)/ws_vdefaults/'
            , bv.WorkspaceVersionDefaultsView.as_view(), name='ws_vdefaults'),
    re_path(r'^vdefaults/'
            , bv.WorkspaceVersionDefaultsView.as_view(), name='vdefaults'),
    re_path(r'^(?P<ws_id>[0-9]+)/jobsum/', bv.jobsum, name='jobsum'),
    re_path(r'^(?P<ws_id>[0-9]+)/job_detail/', bv.job_detail, name='job_detail'),
    re_path(r'^(?P<ws_id>[0-9]+)/jobs/', bv.jobs, name='jobs'),
    re_path(r'^$', bv.index, name='index'),
    re_path(r'^logout/$', av.LogoutView.as_view(), {'next_page': '/account/login/'}, name='mysite_logout'),
    re_path(r'^pwchange/', bv.PasswordChangeView.as_view(success_url='/'), name='pwchange'),
    re_path(r'^publish/(?P<path>.+)', bv.publish),
    re_path(r'^pstatic/(?P<path>.+)', bv.protected_static),
    re_path(r'^api/prot_search/$', ba.prot_search, name='prot_search_api'),
    re_path(r'^api/global_data_prot_search/$', ba.global_data_prot_search, name='global_data_prot_search_api'),
    re_path(r'^api/uniprot/(?P<uniprot>.+)/$', ba.uniprot_lookup, name='uniprot_lookup_api'),
    re_path(r'^api/wsa/(?P<wsa_id>.+)/$', ba.wsa_lookup, name='wsa_lookup_api'),
    re_path(r'^api/list_workspaces/$', ba.list_workspaces, name='list_workspaces_api'),
    re_path(r'^api/list_jobs/$', ba.list_jobs, name='list_jobs_api'),
    re_path(r'^api/ws_molsets/(?P<ws_id>.+)/$', ba.ws_molsets, name='ws_molsets_api'),
    re_path(r'^api/ws_protsets/(?P<ws_id>.+)/$', ba.ws_protsets, name='ws_protsets_api'),
    re_path(r'^api/molset/(?P<ws_id>.+)/(?P<molset_id>.+)/$', ba.molset, name='molset_api'),
    re_path(r'^api/protset/(?P<ws_id>.+)/(?P<protset_id>.+)/$', ba.protset, name='protset_api'),
    re_path(r'^api/search_drugs/(?P<name>.+)/$', ba.search_drugs, name='search_drugs_api'),
    re_path(r'^api/search_wsas/(?P<ws_id>.+)/(?P<name>.+)/$', ba.search_wsas, name='search_wsas_api'),
    re_path(r'^api/fetch_scores/(?P<ws_id>.+)/(?P<job_id>.+)/$', ba.fetch_scores, name='fetch_scores_api'),
    re_path(r'^api/indirect_targets/(?P<target_list>.+)/$', ba.indirect_targets, name='indirect_targets_api'),
    re_path(r'^api/pathway_data/$', ba.pathway_data, name='pathway_data_api'),
    re_path(r'^api/validate_score_json/$', ba.validate_score_json, name='validate_score_json_api'),
    re_path(r'^(?P<ws_id>[0-9]+)/ws_hits/$', bv.WsHitsView.as_view(), name='ws_hits'),
    re_path(r'^hits/$', bv.HitsView.as_view(), name='hits'),
    re_path(r'^selectabilitymodel/(?P<job_id>[0-9]+)/$' , bv.SelectabilityModelView.as_view(), name='selectabilitymodel'),
    re_path(r'^retrospective/$', bv.RetrospectiveView.as_view(), name='retrospective'),
    re_path(r'^suitability/$', bv.SuitabilityView.as_view(), name='suitability'),
    re_path(r'^selectabilityfeatureplot/$', bv.SelectabilityFeaturePlot.as_view(), name='selectabilityfeatureplot'),
    re_path(r'^metricscatterplot/$', bv.MetricScatterPlot.as_view(), name='metricscatterplot'),
    re_path(r'^xws_score_plot/$', bv.XwsScorePlots.as_view(), name='xws_score_plot'),
    re_path(r'^dashboard/$', bv.DashboardView.as_view(), name='dashboard'),
    re_path(r'^(?P<ws_id>[0-9]+)/protset_view/$',
            bv.ProtsetView.as_view(),
            name='protset_view',
            )
]

from django.conf import settings
if False and settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
