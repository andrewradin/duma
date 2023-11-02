from django import template
from browse.utils import extract_list_option
from django.utils.safestring import mark_safe

register = template.Library()

@register.simple_tag(takes_context=True)
def url_add_query(context, **kwargs):
    '''Render a URL to the current page, with an altered query string.

    The string returned contains only the querystring part, which the
    browser interprets as a return the the current page.  The initial
    querystring info is loaded from request.GET, and then altered as
    specified by the passed parameters.
    '''
    request = context.get('request')
    from dtk.duma_view import qstr
    return qstr(request.GET,**kwargs)

@register.simple_tag(takes_context=True)
def view_url(context, **kwargs):
    '''Like above, but for DumaView.'''
    return context.get('view').here_url(**kwargs)

# The following tags provide alternative ways to render "enum" options,
# where a querystring has a small integer value that selects one of a number
# of distinct (named) possibilities.  If these are encapsulted in an Option
# class (as defined in browse/utils.py), and an instance of this class is
# passed in the context, the template can invoke one of the following tags
# to both indicate the selected value, and let the user choose among other
# available values.
#
# option_links can be invoked from the template to create a set of links
# for all values of the query param.  The currently-selected value is
# rendered specially.
#
# option_buttons renders the same information using bootstrap buttons instead
# of links
@register.simple_tag(takes_context=True)
def option_links(context, opt_class, **kwargs):
    if not opt_class:
        return ''
    options = []
    for opt in opt_class.options:
        if opt_class.is_selected(opt):
            options.append(opt_class.label_of(opt))
        else:
            tmp = kwargs.copy()
            tmp[opt_class.parm_name] = opt_class.qparm_val_of(opt)
            options.append(
                '<a href="'
                + url_add_query(context,**tmp)
                + '">'
                + opt_class.label_of(opt) + '</a>'
                )
    return mark_safe("&nbsp;&nbsp;&nbsp;&nbsp;".join(options))

@register.simple_tag(takes_context=True)
def option_buttons(context, opt_class, **kwargs):
    if not opt_class:
        return ''
    options = []
    for opt in opt_class.options:
        tmp = kwargs.copy()
        tmp[opt_class.parm_name] = opt_class.qparm_val_of(opt)
        options.append(
            '<a class="btn btn-default btn-sm'
            + (' btn-primary disabled' if opt_class.is_selected(opt) else '')
            + '" href="' + url_add_query(context,**tmp)
            + '">'
            + opt_class.label_of(opt) + '</a>'
            )
    return mark_safe('<div class="btn-group" role="group">'
            + "".join(options)
            + '</div>'
            )

