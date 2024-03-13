def output_warnings(request,warnings,max_warnings = 20,overflow = 5):
    if len(warnings) > max_warnings + overflow:
        suppressed = len(warnings) - max_warnings
        warnings = warnings[:max_warnings] + [
                f'{suppressed} additional warnings not displayed',
                ]
    from django.contrib import messages
    # use join() to avoid clobbering any safe html in warnings
    from dtk.html import join
    for w in warnings:
        messages.add_message(request, messages.INFO, join('Warning: ',w))

def get_scoreset_warning_summary(ws,ss):
    '''Return a list of each scoreset job with warnings.'''
    from runner.process_info import JobInfo
    from dtk.html import link,glyph_icon,join
    result = []
    for ss_job in ss.scoresetjob_set.all():
        ss_bji = JobInfo.get_bound(ws,ss_job.job_id)
        wlist = ss_bji.get_warnings()
        if wlist:
            # XXX ss_bji.role_label() is another possible fallback, but
            # XXX maybe job_type is more concise and readable
            label = ss_job.label or ss_job.job_type
            result.append(join(
                    f'{len(wlist)} warning(s) for {label} ',
                    link(
                            glyph_icon('link'),
                            ws.reverse('nav_progress',ss_job.job_id),
                            ),
                    ))
    return result
