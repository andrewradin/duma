from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse
from dtk.duma_view import DumaView,list_of,boolean
from dtk.prot_map import DpiMapping
from django import forms
from tools import Enum



class CustomDpiView(DumaView):
    template_name='wsadmin/custom_dpi_view.html'
    GET_parms={
            }

    button_map={
            'create':['dpi'],
            'deprecate': [],
            }

    def make_dpi_form(self,data):
        from wsadmin.models import CustomDpi
        from wsadmin.custom_dpi import CustomDpiModes
        class MyForm(forms.Form):
            dpi = forms.ChoiceField(
                label = 'Base DPI',
                # Prevent custom dpi on top of custom dpi
                choices = DpiMapping.choices(ws=None),
                )
            protset = forms.ChoiceField(
                    label='Protset',
                    choices = self.ws.get_uniprot_set_choices(auto_dpi_ps=False),
                    required=True,
                 )
            
            mode = forms.ChoiceField(
                label = 'Mode',
                choices = CustomDpiModes.choices(),
                help_text= "Subtract removes protset from dpi.  Exclusive filters dpi to keep only prots from protset."
            )
            name = forms.CharField(
                label='Name',
                help_text='Human-friendly name (will appear in DPI selector)',
            )
            descr = forms.CharField(
                label='Description',
                widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                help_text='Extra notes or context to note about this DPI',
                required=False,
            )
        return MyForm(data)

    def create_post_valid(self):
        p = self.context['dpi_form'].cleaned_data

        from .custom_dpi import create_custom_dpi
        create_custom_dpi(
            base_dpi=p['dpi'],
            protset=p['protset'],
            mode=p['mode'],
            name=p['name'],
            descr=p['descr'],
            user=self.request.user.username,
            ws=self.ws,
            )

        return HttpResponseRedirect(self.here_url())
    
    def deprecate_post_valid(self):
        d = self.request.POST
        dpi_id = d['dpi_id']
        action = d['action']

        from wsadmin.models import CustomDpi
        obj = CustomDpi.objects.get(pk=dpi_id)
        obj.deprecated = (action == 'deprecate')
        obj.save()
        return HttpResponseRedirect(self.here_url())
        

    def custom_context(self):
        from .models import CustomDpi
        from .custom_dpi import CustomDpiModes
        rows = CustomDpi.objects.filter(ws=self.ws).order_by('-id')

        from dtk.table import Table
        from django.utils.safestring import mark_safe
        from django.middleware.csrf import get_token
        csrf_token = get_token(self.request)
        csrf = f'<input type="hidden" name="csrfmiddlewaretoken" value="{csrf_token}" />'

        def depr(cust_dpi):
            if cust_dpi.deprecated:
                action = 'undeprecate'
                display = 'Undeprecate'
                btn_class = 'btn-success'
            else:
                action = 'deprecate'
                display = 'Deprecate'
                btn_class = 'btn-danger'
            content = f'''
            <form method='POST'>
                {csrf}
                <input type='hidden' name='dpi_id' value='{cust_dpi.id}' />
                <input type='hidden' name='action' value='{action}' />
                <button class='btn btn-sm {btn_class}' name='deprecate_btn'>{display}</button>
            </form>
            '''
            
            return mark_safe(content)

        columns = [
            Table.Column('name'),
            Table.Column('descr'),
            Table.Column('base_dpi'),
            Table.Column('mode', cell_fmt=lambda x: CustomDpiModes.get('label', x)),
            Table.Column('prot_set', cell_fmt=lambda x: self.ws.get_uniprot_set_name(x)),
            Table.Column('uid'),
            Table.Column('created_by'),
            Table.Column('created_on'),
            Table.Column('protset_prots'),
            Table.Column('base_prots'),
            Table.Column('final_prots'),
            Table.Column('base_molecules'),
            Table.Column('final_molecules'),
            Table.Column('Deprecate', extract=depr),
        ]

        self.context_alias(
            table=Table(rows, columns)
        )