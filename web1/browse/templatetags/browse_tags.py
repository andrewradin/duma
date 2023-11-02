from django import template
from tools import percent
from browse.views import is_demo
from browse.models import WsAnnotation,Vote

register = template.Library()

from tools import sci_fmt
register.filter('sci_fmt',sci_fmt)

@register.simple_tag(takes_context=True)
def duma_menu(context,menu_name):
    from web1.menu import build_menu
    return build_menu(menu_name,context)

@register.simple_tag(takes_context=True)
def disposition_form(context,wsa):
    rff_key = 'reclassify_form_factory'
    if rff_key in context:
        rff = context[rff_key]
    else:
        from nav.views import ReclassifyFormFactory
        rff = ReclassifyFormFactory()
        context[rff_key] = rff
    eform_key = 'election_form'
    if eform_key in context:
        top = context[eform_key].flavor.top()
    else:
        top=(
                wsa.indication_vals.INACTIVE_PREDICTION,
                wsa.indication_vals.PATENT_PREP,
                )
    FormClass = rff.get_form_class(wsa,top=top)
    form = FormClass()
    from django.utils.html import format_html
    import dtk.html as wgt
    return format_html('{}<br>ind href:{}<br>{}{}',
            form['indication'],
            form['indication_href'],
            wgt.button('Update from form',name='reclassify',
                    extra_attrs={'class':'btn btn-primary'},
                    ),
            form['wsa_id'],
            )

@register.simple_tag(takes_context=True)
def protlink(context,uniprot):
    # This can only be used on pages where uniprot2gene_map has been
    # pre-loaded into the context
    map_name='uniprot2gene_map'
    from dtk.html import link
    return link(
            context[map_name].get(uniprot,'('+uniprot+')'),
            context['ws'].reverse('protein',uniprot),
            )

@register.filter
def lookup(obj, value):
    return obj[value]


@register.filter
def safe_json(obj):
    from django.utils.safestring import mark_safe
    import json
    from dtk.plot import convert
    return mark_safe(json.dumps(obj, default=convert))

@register.simple_tag(takes_context=True)
def bulk_update_links(context,prefix):
    import dtk.html
    return dtk.html.bulk_update_links(prefix)

@register.simple_tag(takes_context=True)
def pad(context):
    import dtk
    return dtk.html.pad()

@register.filter
def path_header(value):
    (lvl,fld) = value.split(':')
    return " ".join((lvl.upper(),fld))

@register.filter
def format_str(value,fmt):
    return fmt % value

@register.simple_tag
def plotly_js_file():
    import plotly.offline as po
    from path_helper import PathHelper
    import os
    ver = po.get_plotlyjs_version()
    fn = f'plotly.{ver}.js'
    path = os.path.join(PathHelper.publish, 'js', fn)
    if not os.path.exists(path):
        dr = os.path.dirname(path)
        if not os.path.exists(dr):
            os.makedirs(dr, exist_ok=True)
        from atomicwrites import atomic_write
        with atomic_write(path, overwrite=True) as f:
            f.write(po.get_plotlyjs())
    return fn

@register.simple_tag(takes_context=True)
def needs_this_vote(context,wsa):
    user = context['request'].user
    if Vote.needs_this_vote(wsa,user):
        import dtk.html as wgt
        return wgt.glyph_icon('arrow-right',color='firebrick')
    return ''

@register.simple_tag
def glyph_icon(name,**kwargs):
    import dtk.html as wgt
    return wgt.glyph_icon(name,**kwargs)

@register.simple_tag
def info_icon(text,**kwargs):
    import dtk.html as wgt
    return wgt.glyph_icon('info-sign',hover=text,**kwargs)

@register.simple_tag(takes_context=True)
def note_icon(context,text,**kwargs):
    if is_demo(context['user']):
        text = 'Note text suppressed in demo mode'
    import dtk.html as wgt
    return wgt.glyph_icon('comment',hover=text,**kwargs)

@register.simple_tag
def bool_icon(value):
    import dtk.html as wgt
    if value:
        return wgt.glyph_icon('ok',color='green')
    return wgt.glyph_icon('remove',color='firebrick')

@register.simple_tag
def vote_icon(user,vote):
    import dtk.html as wgt
    if vote is None or vote.recommended is None:
        return wgt.glyph_icon('question-sign',color='goldenrod',hover=user)
    if vote.recommended:
        return wgt.glyph_icon('ok',color='green',hover=user)
    return wgt.glyph_icon('remove',color='firebrick',hover=user)

@register.simple_tag
def my_vote_note(user, vote):
    from dtk.html import note_format
    return note_format(vote.get_note_text(str(user)))

@register.simple_tag(takes_context=True)
def infer_title(context):
    remap = {
        'indications':'database',
        'settings/0':'paths',
        'ml':'classifier',
        }
    label = context['function_root']
    if label in remap:
        label = remap[label]
    if 'ws' in context:
        label += "(%d)" % context['ws'].id
    return label

@register.simple_tag(takes_context=True)
def pathfield(context,field,counter):
    col_name = context['header'][int(counter)]
    (lvl,fld) = col_name.split(':')
    if fld in ('protein','prot2'):
        tmpl = template.Template("{% include 'browse/prot_link.html' %}")
        c = {'prot_id':field,'ws':context['ws']}
        return tmpl.render(template.Context(c))
    if fld == 'tissue':
        try:
            return context['tissue_map'][int(field)]
        except KeyError:
            return "(deleted tissue %s)" % field
    return field

@register.simple_tag(takes_context=True)
def drugname(context,drug_ws):
    return drug_ws.get_name(is_demo(context['user']))

@register.filter(name='is_button_pusher')
def is_button_pusher(user):
    return user.groups.filter(name='button_pushers').count() > 0

@register.inclusion_tag('workflow_status_button.html', takes_context=True)
def workflow_status_button(context, id):
    ws = context['ws']
    from stages import WorkflowStage
    stage = WorkflowStage.get_or_create_obj(ws, id)
    return {
            'id': id,
            'button_classes': WorkflowStage.button_classes_for_status(stage.status),
            'status_text': WorkflowStage.status_text_for_status(stage.status),
            'ws': ws,
            }

@register.simple_tag()
def attr_summary(drug):
    l = []
    if drug.approved:
        l.append('Approved')
    if drug.experimental:
        l.append('Experimental')
    if drug.illicit:
        l.append('Illicit')
    if drug.investigational:
        l.append('Investigational')
    if drug.neutraceutical:
        l.append('Neutraceutical')
    if drug.withdrawn:
        l.append('Withdrawn')
    if drug.hide:
        l.append('Hidden')
    return ", ".join(l)
