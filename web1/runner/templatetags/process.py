from django import template
from django.utils.html import escape
import os

register = template.Library()

from runner.models import Process,Load
from algorithms.exit_codes import ExitCoder
from runner.process_info import JobCrossChecker
from django.utils.safestring import mark_safe

@register.filter()
def job_status(process):
    enum = Process.status_vals
    if process.status == enum.QUEUED:
        behind = [ x for x in process.wait_for.all() ]
        if behind == []:
            return "Ready"
        return ("Waiting for "
                +" ".join([ x.name+"("+str(x.id)+")" for x in behind ])
                )
    return enum.get('label',process.status)

@register.simple_tag(takes_context=True)
def job_page(context,job_id):
    ws = context['ws']
    from dtk.html import link,glyph_icon
    return link(
        glyph_icon('link'),
        ws.reverse('nav_progress',job_id)
        )

@register.simple_tag()
def job_log_tail(process):
    """Returns the end of the job log, if available"""
    # Number of bytes from the end of the file to read, at maximum.
    BYTES_TO_READ = 4096
    path = process.logfile()
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                # Find the size of the file.
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read the last N bytes, or the whole thing if N > filesize.
                f.seek(-min(size, BYTES_TO_READ), os.SEEK_END)
                out = f.read()
            # Need non-strict errors - because we're grabbing by bytes, it's possible
            # that we're truncating a multibyte character either at the start or end of the chunk,
            # and what's left over may not be valid utf8.
            out = out.decode('utf8', errors='backslashreplace')
            return out
    except IOError as e:
        return str(e)

    return ''


@register.filter()
def job_times(process):
    from django.utils.html import format_html
    from dtk.text import fmt_time,fmt_delta,fmt_timedelta
    from django.utils import timezone
    from datetime import timedelta
    now = timezone.now()
    if process.completed:
        return format_html(
            u'<br>completed {}<br>ran {}<br>queued {}<br>waited {}',
            fmt_time(process.completed),
            fmt_delta(process.completed,process.started),
            fmt_delta(process.started,process.created),
            fmt_timedelta(process.waited()),
            )
    if process.started:
        return format_html(
            u'<br>started {}<br>active {} and counting<br>queued {}<br>waited {}',
            fmt_time(process.started),
            fmt_delta(now,process.started),
            fmt_delta(process.started,process.created),
            fmt_timedelta(process.waited()),
            )
    if process.created:
        return format_html(
            u'<br>created {}<br>queued {} and counting',
            fmt_time(process.created),
            fmt_delta(now,process.created),
            )

@register.filter()
def job_date(process):
    fmt = "%m/%d/%y, %I:%M%p"
    from dtk.text import fmt_time
    if process.completed:
        return fmt_time(process.completed,fmt)
    if process.started:
        return fmt_time(process.started,fmt) + " (start time)"
    return fmt_time(process.created,fmt) + " (create time)"

@register.simple_tag(takes_context=True)
def make_protset(context,dc,code):
    ws = context['ws']
    label = dc.get_label(code)
    from dtk.url import UrlConfig
    url = UrlConfig(ws.reverse('nav_ps'))
    prot_list = dc.get_keyset(code)
    from dtk.html import link
    return link(label,url.here_url({
                        'uniprots': ','.join(prot_list),
                        })
                )

@register.simple_tag(takes_context=True)
def job_ws(context,process):
    try:
        # As a potential performance optimization on pages listing many
        # jobs, the view can pass a single JobCrossChecker in the context
        # that will be used for all jobs.
        jcc_key = 'job_cross_checker'
        jcc = context[jcc_key]
    except KeyError:
        jcc = JobCrossChecker()
    return mark_safe('<br>'.join(jcc.job_ws(process.name)))

@register.simple_tag(takes_context=True)
def job_summary(context,process,flavor=''):
    return job_summary_impl(context,process,flavor)

def job_summary_impl(context,process,flavor=''):
    # The flavor parameter allows templates to pass a single string of
    # space-separated options names, possibly preceeded by '-',
    # that will change any optional processing from the default.
    # If existing processing is made optional, it can be added to the
    # default list below so that templates only need updating if we want
    # their behavior to change.
    options = set( "status settings outlink".split() )
    for mod in flavor.split():
        if mod[0] == '-':
            options.remove(mod[1:])
        else:
            options.add(mod)
    # guard against null job records
    if not process:
        return 'No Job Record'
    placement='bottom' if 'bottom' in options else 'top'
    try:
        # As a potential performance optimization on pages listing many
        # jobs, the view can pass a single JobCrossChecker in the context
        # that will be used for all jobs.
        jcc_key = 'job_cross_checker'
        jcc = context[jcc_key]
    except KeyError:
        jcc = JobCrossChecker()
    items=[]
    enum = Process.status_vals
    from dtk.html import link,glyph_icon
    if 'status' in options:
        # pre-set default status and info text
        es = jcc.extended_status(process)
        status = es.label
        info_txt = None
        if status == 'OutOfDate':
            info_txt = 'BEHIND: '+' '.join(es.newer)
        elif process.status == enum.FAILED and process.exit_code:
            ec = ExitCoder()
            info_txt = ec.message_of_code(process.exit_code)
        elif process.status == enum.QUEUED:
            # annotate waiting jobs as in job_status() above
            behind = [ x for x in process.wait_for.all() ]
            if behind == []:
                info_txt = "awaiting resources"
            else:
                info_txt = ("Waiting for "
                    +" ".join([ x.name+"("+str(x.id)+")" for x in behind ])
                    )
        # store status, along with any detail
        items.append(status)
        if info_txt:
            items.append(glyph_icon('info-sign',
                        hover=escape(info_txt),
                        placement=placement,
                        ))
    if 'settings' in options:
        # settings if available
        settings_string=', '.join(process.settings_json.split(','))
        items.append(glyph_icon('cog',
                    hover=escape(settings_string),
                    placement=placement,
                    ))
    # note if available
    if process.note:
        items.append(glyph_icon('comment',
                    hover=escape(process.get_note_text()),
                    placement=placement,
                    ))
    # output page link, if we're in the workspace the job belongs to
    if 'outlink' in options:
        try:
            workspaces = list(jcc.job_ws_obj(process.name, context['ws']))
            # There are two cases where we get multiple workspaces for a job, tissues and cross-ws models.
            # If you're already inside a workspace, or a job was run with a specific workspace in its settings,
            # default to only showing those.
            # Otherwise (which only happens for viewing tissue jobs on the jobsum page) we can just list all.
            if len(workspaces) > 1:
                if context['ws']:
                    workspaces = [context['ws']]
                elif 'ws_id' in process.settings():
                    from browse.models import Workspace
                    workspaces = list(Workspace.objects.filter(pk=process.settings()['ws_id']))
            
            # I believe this should only happen if the CM is hidden and process_info doesn't know about it.
            if len(workspaces) == 0 and context['ws']:
                workspaces = [context['ws']]

            for ws in workspaces:
                items.append(link(
                            glyph_icon('link'),
                            ws.reverse('nav_progress',process.id),
                            ))
        except KeyError:
            pass
    # log link if file exists
    from dtk.html import join
    if 'log' in options:
        path = process.logfile()
        if os.path.exists(path):
            from path_helper import PathHelper
            items.append(join(
                    '(',
                    link("log",PathHelper.url_of_file(path)),
                    ')',
                    sep='',
                    ))
    rm_key = 'rm'
    if rm_key in context:
        rm = context[rm_key]
        try:
            desc = rm.desc(process.id)
            if desc:
                items.append(desc)
        except Load.DoesNotExist:
            pass
    return join(*items)

@register.simple_tag(takes_context=False)
def radio_cell(group,choices):
    import dtk.html
    return dtk.html.radio_cell(group,choices)

@register.simple_tag(takes_context=True)
def start_stop_button(context,mch):
    if not mch.has_buttons:
        return ""
    user = context['request'].user
    if not user.groups.filter(name='button_pushers').count():
        return ""
    i = mch.get_ec2_instance()
    templ = '<input type="submit" name="%s" value="%s"/>'
    if i.state == 'stopped':
        return mark_safe(templ % ('start','Start'))
    if i.state == 'running':
        return mark_safe(templ % ('stop','Stop'))
    return ""

@register.simple_tag(takes_context=True)
def change_type_buttons(context,mch):
    if not mch.has_buttons:
        return ""
    user = context['request'].user
    if not user.groups.filter(name='button_pushers').count():
        return ""
    downgrade_type,upgrade_type = mch.get_adjacent_types()
    result = ""
    templ = '<input type="submit" name="%s" value="%s"/>'
    if downgrade_type:
        result += templ % ('downgrade',u'\u25bc '+downgrade_type)
    if upgrade_type:
        result += templ % ('upgrade',u'\u25b2 '+upgrade_type)
    return mark_safe(result)

@register.simple_tag(takes_context=True)
def job_clear_controls(context):
    user = context['request'].user
    if not user.groups.filter(name='button_pushers').count():
        return ""
    return mark_safe('''
        <input class="button btn-danger"
                type="submit" name="killall" value="Abort all active jobs"/>
        <input type="checkbox" name="check"/>Yes, really!
    ''')
