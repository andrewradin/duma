from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

import logging
logger = logging.getLogger(__name__)

class CreateWsView(DumaView):
    template_name='wsmgr/create_ws.html'
    index_dropdown_stem='wsmgr:create_ws'
    button_map={
            'create':['spec'],
            }
    def custom_setup(self):
        if not self.in_group('duma_admin'):
            return HttpResponseRedirect('/')
    def make_spec_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        ff.add_field('name', forms.CharField(
                label = "Workspace Name",
                max_length = 256,
                ))
        ff.add_field('active',forms.BooleanField(
                label = 'Include in Active list',
                required = False,
                initial = True,
                ))
        ff.add_field('cross_compare',forms.BooleanField(
                label = 'Include in Cross-compare',
                required = False,
                initial = True,
                ))
        ff.add_field('src_ws_id', forms.IntegerField(
                label = 'Copy From Workspace ID',
                required = False,
                ))
        from dtk.copy_ws import add_copy_options
        add_copy_options(ff,True)
        FormClass = ff.get_form_class()
        return FormClass(data)
    def create_post_valid(self):
        p = self.spec_form.cleaned_data
        from browse.models import Workspace
        if Workspace.objects.filter(name=p['name']).exists():
            self.spec_form.add_error(
                    'name',
                    "This name is already in use.",
                    )
            return
        dst_ws = Workspace.objects.create(**{
                x:p[x] for x in ('name','active','cross_compare')
                })
        if p['src_ws_id']:
            src_ws = Workspace.objects.get(pk=p['src_ws_id'])
            from dtk.copy_ws import do_copies
            do_copies(self.username(),p,src_ws,dst_ws)
        return HttpResponseRedirect(dst_ws.reverse('workflow'))

class CopyWsView(DumaView):
    template_name='wsmgr/copy_ws.html'
    index_dropdown_stem='wsmgr:copy_ws'
    button_map={
            'copy':['spec'],
            }
    def custom_setup(self):
        if not self.in_group('duma_admin'):
            return HttpResponseRedirect(self.ws.reverse('workflow'))
    def make_spec_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        ff.add_field('src_ws_id', forms.IntegerField(
                label = 'Source Workspace ID',
                ))
        from dtk.copy_ws import add_copy_options
        add_copy_options(ff,False)
        FormClass = ff.get_form_class()
        return FormClass(data)
    def copy_post_valid(self):
        p = self.spec_form.cleaned_data
        if p['src_ws_id'] == self.ws.id:
            self.spec_form.add_error(
                    'src_ws_id',
                    "You can't copy from the workspace you're in.",
                    )
            return
        from browse.models import Workspace
        src_ws = Workspace.objects.get(pk=p['src_ws_id'])
        from dtk.copy_ws import do_copies
        do_copies(self.username(),p,src_ws,self.ws)
        return HttpResponseRedirect(self.ws.reverse('workflow'))

class ImportHistoryView(DumaView):
    template_name='wsmgr/imphist.html'
    GET_parms={
            'sort':(SortHandler,'-timestamp'),
            }
    def custom_context(self):
        from .models import ImportAudit
        qs = ImportAudit.objects.filter(ws=self.ws)
        # start with default ordering, then hack in special cases
        order = [self.sort.to_string()]
        if self.sort.colspec == 'collection':
            # collection sort orders by name
            order[0] += '__name'
        if self.sort.colspec != 'timestamp':
            # non-timestamp sorts do newest-to-oldest minor sort on timestamp
            order.append('-timestamp')
        qs = qs.order_by(*order)
        from dtk.table import Table
        from dtk.text import fmt_time
        self.context['table'] = Table(qs,[
                Table.Column('Timestamp',
                        cell_fmt=fmt_time,
                        sort='h2l',
                        ),
                Table.Column('Operation',
                        ),
                Table.Column('Collection',
                        sort='l2h',
                        ),
                Table.Column('Molecule',
                        extract=lambda x: x.wsa.html_url() if x.wsa else '',
                        ),
                Table.Column('Collection Version',
                        code='coll_ver',
                        ),
                Table.Column('Cluster Version',
                        code='clust_ver',
                        ),
                Table.Column('User',
                        sort='l2h',
                        ),
                Table.Column('Succeeded',
                        ),
                ],
                url_builder=self.here_url,
                sort_handler=self.sort,
                )
