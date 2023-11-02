from django import forms
from django.http import HttpResponseRedirect

from dtk.duma_view import DumaView

class FlagSetView(DumaView):
    template_name='flagging/flagset.html'
    index_dropdown_stem='flagset'
    button_map = {
            'update':['enables']
            }
    def custom_setup(self):
        from .models import FlagSet
        self.flagsets=FlagSet.objects.filter(ws=self.ws).order_by('-id')
    def make_enables_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for item in self.flagsets:
            field_code = 'flagset_%d' % item.id
            ff.add_field(field_code,forms.BooleanField(
                                            initial=item.enabled,
                                            required=False,
                                            ))
            item.field_code = field_code
        return ff.get_form_class()(data)
    def update_post_valid(self):
        p = self.context['enables_form'].cleaned_data
        for item in self.flagsets:
            new_val = p[item.field_code]
            if new_val != item.enabled:
                item.enabled = new_val
                item.save()
        return HttpResponseRedirect('#')
    def custom_context(self):
        from dtk.table import Table
        from dtk.text import fmt_time
        from dtk.html import link
        from dtk.duma_view import qstr
        enables_form = self.context['enables_form']
        self.context_alias(table=Table(self.flagsets,[
                Table.Column('Source'),
                Table.Column('Settings'),
                Table.Column('Drugs Flagged',
                        extract=lambda x:link(
                                str(x.drug_count()),
                                self.ws.reverse('rvw:review')+qstr({},
                                        flavor='flagset_%d'%x.id,
                                        )
                                )
                        ),
                Table.Column('Enabled',
                        extract=lambda x:enables_form[x.field_code],
                        ),
                Table.Column('Created',
                        cell_fmt=fmt_time,
                        ),
                ]))
