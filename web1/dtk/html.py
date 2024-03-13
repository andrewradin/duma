def attr_text(attr):
    from django.utils.html import format_html_join
    return format_html_join(' ','{}="{}"',attr.items())

def tag_wrap(tag,content,attr=None):
    from django.utils.html import format_html, conditional_escape
    from django.utils.safestring import mark_safe
    attrs = attr_text(attr) if attr else ''
    if attrs:
        attrs = mark_safe(' '+attrs)
    if content is None:
        return format_html('<{}{}/>',
                tag,
                attrs,
                )
    # attrs are already passed through format_html above.
    # tag shouldn't be user-provided, so can be directly included.
    return mark_safe(f'<{tag}{attrs}>{conditional_escape(content)}</{tag}>')

def alert(txt):
    return tag_wrap('font',txt,{'color':'firebrick'})

def join(*args,**kwargs):
    sep = kwargs.get('sep',' ')
    from django.utils.html import format_html_join
    return format_html_join(sep,u'{}',((x,) for x in args))

def ulist(parts):
    return tag_wrap('ul',join(*[
            tag_wrap('li',part)
            for part in parts
            ]))

def link(text,href,new_tab=False, extra_attrs={}):
    attrs = {**extra_attrs, 'href':href}
    if new_tab:
        attrs['target'] = '_blank'
        from django.utils.html import format_html
        text = format_html('{}&nbsp;{}',text,glyph_icon('new-window'))
    return tag_wrap('a',text,attrs)

def nowrap(text,tag='span'):
    return tag_wrap(tag,text,{'style':'white-space:nowrap'})

def pad(sp='6pt'):
    return tag_wrap('span',' ',{'style':'padding-right:'+sp})

def pad_table(header,rows):
    from django.utils.html import format_html_join,format_html
    header_fmt = format_html_join('', u'<th>{}</th><th>{}</th>', (
                ( x, pad() )
                for x in header
                ))
    rows_fmt = [
            format_html_join('', u'<td>{}</td><td></td>', (
                    (col,)
                    for col in row
                    ))
            for row in rows
            ]
    return tag_wrap('table',
            format_html(u'<tr>{}</tr>{}',
                    header_fmt,
                    format_html_join('',u'<tr>{}</tr>', (
                            ( x, )
                            for x in rows_fmt
                            )),
                    ),
            )

def rows_from_dict(d,l):
    # d is a dictionary
    # l is a list of (label,key_for_d)
    # returns a list of [label,value_from_d_or_NA] of the same length as l
    rows = []
    from tools import sci_fmt
    for label,key in l:
        v = 'N/A'
        try:
            v = sci_fmt(d[key])
        except (KeyError,ValueError):
            pass
        rows.append( [label,v] )
    return rows

def table_from_dict(d,l):
    # inputs as above, returns a 2-column html table
    return pad_table(['',''],rows_from_dict(d,l))

def bar_chart_from_dict(d,l,
            bar_width=300,
            bar_colors=('seagreen','darkseagreen'),
            ):
    # inputs as above, returns a 3-column html table, where the 3rd column
    # is a bar chart
    rows = rows_from_dict(d,l)
    for i,row in enumerate(rows):
        v = row[1]
        try:
            px=int(bar_width*float(v))
            color=bar_colors[i % len(bar_colors)]
            bar = tag_wrap('div','.',{
                    'style':'color:%s; background-color:%s; width:%dpx'%(
                            color,color,px
                            ),
                    })
        except ValueError:
            bar=''
        row.append(bar)
    return pad_table(['','',''],rows)

def lines_from_pairs(l):
    '''Return html for multiple lines, one per tuple in input iterable.

    Input tuples are assumed to be (key,value).
    Lines are <br>-separated, in sorted order by key, and each look like
    key:value.
    '''
    if not l:
        return '' # might be None
    from django.utils.safestring import mark_safe
    return join(*[
            k+':'+str(v)
            for k,v in sorted(l,key=lambda x:x[0])
            ],sep=mark_safe('<br>'))

def entry(name,initial='',attr={}):
    attr = dict(attr)
    attr['type'] = 'text'
    attr['name'] = name
    if initial:
        attr['value'] = initial
    return tag_wrap('input','',attr)

def radio_cell(group,choices,container='div',checked=None):
    '''Return HTML for a stack of radio buttons.

    'choices' is an iterable returning tuples of the form (code,label),
    as in a drop-down.  Buttons from multiple calls can be chained
    together if the same 'group' parameter is passed, and the code fields
    are distinct.  This can be used for generating one cell in a table
    where radio behavior spans rows. POST handling is trivial -- the
    value of POST[group] will be the code of the selected button.
    '''
    widget=u'''
        <{} style="white-space:nowrap">
            <input type="radio" name="{}" value="{}" {}>{}</input>
        </{}>
        '''
    from django.utils.html import format_html_join
    return format_html_join('',widget, (
                (
                    container,
                    group,
                    x[0],
                    'checked="checked"' if x[0] == checked else '',
                    x[1],
                    container,
                )
                for x in choices
                ))

def checklist(choices,checked):
    widget=u'''
        <span style="white-space:nowrap; padding-right:6pt">
            {}: <input type="checkbox" name="{}" {}/>
        </span>
        '''
    from django.utils.html import format_html_join
    return format_html_join('',widget, (
                (
                    x[1],
                    x[0],
                    'checked="checked"' if x[0] in checked else '',
                )
                for x in choices
                ))

def bulk_update_links(prefix,attr='name'):
    widget=u'''
        <a
            href="#bulk_update_{}"
            onclick="bulk_update_checks({},'{}','{}');"
            style="white-space:nowrap"
            id='bulk_update_{}'
            >
                {}
        </a>
        '''
    from django.utils.html import format_html_join
    return format_html_join(pad(),widget, (
                (prefix,'false',prefix,attr,prefix,'deselect all'),
                (prefix,'true',prefix,attr,prefix,'select all'),
                ))

def move_to_top(choices,tops):
    result = []
    for x in tops:
        for y in choices:
            if x == y[0]:
                result.append(y)
                break
    for y in choices:
        if y[0] not in tops:
            result.append(y)
    return result

def button(label,name=None,extra_attrs={}):
    attr={'type':'submit'}
    attr.update(extra_attrs)
    if name:
        attr['name'] = name+'_btn'
    return tag_wrap('button',label,attr=attr)

def hidden(name,value):
    widget=u'<input type="hidden" name="{}" value= "{}"/>'
    from django.utils.html import format_html
    return format_html(widget,
            name,
            value,
            )

def dropdown(name,choices,checked,mod_flag='! '):
    select=u'<select name="{}">{}</select>'
    option=u'<option value="{}" {}>{}{}</option>'
    from django.utils.html import format_html_join,format_html
    return format_html(select,
            name,
            format_html_join('',option, (
                    (
                        x[0],
                        'selected' if x[0] == checked else '',
                        '' if x[0] == checked else mod_flag,
                        x[1],
                    )
                    for x in choices
                    ))
            )

def decimal_cell(val,href=None,fmt=None):
    # HTML apparently doesn't have a working decimal tab,
    # so we render a fixed number of spaces right of the
    # decimal, and then right-justify.
    # XXX The web suggested we can enhance this by
    # XXX suppressing trailing zeros (and the decimal point,
    # XXX if appropriate) by wrapping them in <span
    # XXX style="visibility:hidden"></span>.
    from django.utils.html import format_html
    if fmt is None:
        try:
           f_val = abs(float(val))
        except (ValueError,TypeError):
            f_val = None
        if f_val and f_val < 0.1:
            fmt='%.2e'
        else:
            fmt="%0.2f"
    try:
        val = fmt % float(val)
    except (ValueError,TypeError):
        if val is None:
            val = ''
    if href:
        val = link(val,href)
    return format_html(u'<td style="text-align:right">{}</td>',val)

def hover(text,hover):
    return tag_wrap('span',text,{
                'data-toggle':'tooltip',
                'title':hover,
                })

def truncate_hover(text_list, max_len, sep=', ', pop_sep='<br>'):
    # Use +1 so that we only create an expando list if there's 2 or more beyond limit.
    if len(text_list) > max_len + 1:
        visible = sep.join(text_list[:max_len])
        rem = pop_sep.join(text_list[max_len:])
        return visible + sep + popover(f'...[{len(text_list) - max_len} more]', rem)
    else:
        return sep.join(text_list)

# as of PLAT-1782, all references to bootstrap glyphicon HTML go
# through here, except:
# - drag-and-drop row handles
# - the '+' job select icon, which will be removed in PLAT-1776
# - some special case code in selenium that follows a job link
# The above will need to be dealt with when porting to bootstrap 3.2.3
def glyph_icon(name,color=None,hover=None,html=False,placement=None):
    attrs = {
            'class':'glyphicon glyphicon-'+name,
            'aria-hidden':'true',
            }
    if hover:
        attrs['data-toggle'] = 'tooltip'
        attrs['title'] = hover
    if html:
        attrs['data-html'] = 'true'
    if placement:
        attrs['data-placement'] = placement
    icon = tag_wrap('span','',attrs)
    if color:
        icon = tag_wrap('font',icon,{'color':color})
    return icon

def popover(el, content):
    attrs = {
            'data-toggle': 'popover',
            'data-trigger': 'focus',
            'data-html': True,
            'tabindex': '0',
            'role': 'button',
            'style': 'cursor:pointer',
            'data-content': content
        }
    return tag_wrap('a', el, attrs)

def note_format(text):
    from django.utils.html import linebreaks,urlize
    from django.utils.safestring import mark_safe
    return mark_safe(
                linebreaks(
                        urlize(
                                text,
                                trim_url_limit=10,
                                autoescape=True,
                                )
                        )
                )

def tie_icon(ranker,key):
    higher,tied,lower = ranker.get_details(key)
    if tied > 1:
        return glyph_icon('info-sign',hover='%d tied' % tied)
    return ''

# Special widget for demerits, etc.
from django import forms
import six
class WrappingCheckboxSelectMultiple(forms.widgets.CheckboxSelectMultiple):
    template_name = "_label_first_checkbox_group.html"
    option_template_name = "_label_first_checkbox_option.html"
    def format_value(self, value):
        from django.utils.encoding import force_text
        return [force_text(x) for x in value or []]

# Special widget for suppressing browser-side validation of required fields.
# Django 1.11 adds the 'required' attribute to the HTML for required fields.
# This is normally a good thing, but can cause problems when the HTML form
# that the browser sees is actually made up of multiple Django Form objects,
# and has multiple submit buttons, and we want validation of the parts to
# be dependent on which button is used (via a DumaView button_map). In this
# case we may need to suppress the browser-side validation, and wait to
# validate the fields on the server.
class NonRequiredTextInput(forms.widgets.TextInput):
    def use_required_attribute(self,initial):
        return False



class WsaInput(forms.widgets.TextInput):
    """Use this as the widget for an IntegerField"""
    template_name = '_wsa_input.html'

    def __init__(self, ws, *args, **kwargs):
        self.ws = ws
        super().__init__(*args, **kwargs)

    def get_context(self, *args, **kwargs):

        context = super().get_context(*args, **kwargs)

        initial = context['widget'].get('value', None)
        if initial:
            from browse.models import WsAnnotation
            wsa = WsAnnotation.objects.get(pk=context['widget']['value'])
            context['widget']['initial_name'] = wsa.agent.canonical

        context['widget']['ws'] = self.ws
        return context


class MultiField(forms.fields.CharField):
    def __init__(self, jstype, extra_attrs=None, value_setup=None, valtype=None, *args, **kwargs):
        valtype = valtype or str
        widget = MultiInput(jstype=jstype, extra_attrs=extra_attrs, value_setup=value_setup)
        super().__init__(widget=widget, *args, **kwargs)
        self.valtype = valtype

    def clean(self, value):
        value = super().clean(value)
        return self.valtype(value)

class MultiInput(forms.widgets.TextInput):
    template_name = '_multi_input_widget.html'
    def __init__(self, jstype, extra_attrs=None, value_setup=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jstype = jstype
        self.extra_attrs = extra_attrs or {}
        self.value_setup = value_setup

    def get_context(self, *args, **kwargs):
        context = super().get_context(*args, **kwargs)
        if self.value_setup:
            context['widget']['value'] = self.value_setup(context['widget']['value'])
        context['widget']['jstype'] = self.jstype
        context['widget']['extra_attrs'] = self.extra_attrs(context['widget']['value']) if self.extra_attrs else {}
        return context


def parse_wsa_list(valstr):
    if not valstr:
        return []
    if valstr[0] == '[':
        import json
        return json.loads(valstr)
    import re
    wsas = re.split(r'[,\s]+', valstr)
    return [int(x.strip()) for x in wsas if x.strip()]

class MultiWsaField(MultiField):
    """Convenience specialization for multiple WSAs.

    The internals are a bit awkward here, mostly to make this usable as a dropin replacement
    for the textarea-style comma/newline separated wsa lists.
    """
    def __init__(self, ws_id, *args, **kwargs):
        def value_setup(value):
            """Converts from wsas-string to wsas-list"""
            if isinstance(value, str):
                value = parse_wsa_list(value)
            if value is None:
                value = []
            return value

        def extra_attrs(value):
            from browse.models import WsAnnotation
            if value:
                wsas = WsAnnotation.objects.filter(pk__in=value)
            else:
                wsas = []
            id_to_name = {wsa.id: wsa.agent.canonical for wsa in wsas}
            return {'wsId': ws_id, 'idToName': id_to_name}
        super().__init__(jstype='MolSearch', extra_attrs=extra_attrs, value_setup=value_setup, *args, **kwargs)

    def clean(self, value):
        value = super().clean(value)
        import json
        value = json.loads(value)
        # Everything expects comma separated wsas
        return ','.join(str(x) for x in value)
