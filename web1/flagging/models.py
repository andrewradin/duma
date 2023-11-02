from django.db import models

class FlagSet(models.Model):
    ws = models.ForeignKey('browse.Workspace', on_delete=models.CASCADE)
    source = models.CharField(max_length=100)
    settings = models.TextField(blank=True,default='')
    created = models.DateTimeField(auto_now_add=True)
    enabled = models.BooleanField(default=True)
    def drug_count(self):
        return self.flag_set.values('wsa_id').distinct().count()

class Flag(models.Model):
    wsa = models.ForeignKey('browse.WsAnnotation', on_delete=models.CASCADE)
    run = models.ForeignKey(FlagSet, on_delete=models.CASCADE)
    category = models.CharField(max_length=100)
    detail = models.CharField(max_length=256,default='',blank=True)
    href = models.TextField(blank=True,default='')
    new_tab = True # hook for format_flags
    @classmethod
    def get_for_wsa(cls,wsa):
        sets = FlagSet.objects.filter(ws=wsa.ws,enabled=True)
        flags = cls.objects.filter(wsa=wsa,run__in=sets)
        return list(flags.order_by('category'))
    @classmethod
    def format_flags(cls,flags,condense=True):
        from dtk.html import link
        from django.utils.html import format_html_join
        if condense:
            filtered_flags = []
            most_recent_prevTarg = max(
                                       [0]+[
                                            x.id
                                            for x in flags
                                            if x.category == 'PreviousTargets'
                                            ])
            for flag in flags:
                if flag.id == most_recent_prevTarg or flag.category != 'PreviousTargets':
                    filtered_flags.append(flag)
            flag_html = format_html_join('',u'<b>{}:</b> {}<br>',
                [('Flags','')] + sorted(list(set([
                        (x.category,link(
                                        ','.join(sorted(x.detail.replace('+', '-').split(','))) or 'link',
                                        x.href,
                                        new_tab=x.new_tab,
                                        ))
                        for x in filtered_flags
                        ])))
                )
        else:
            flag_html = format_html_join('',u'<b>{}:</b> {}<br>',
                [('Flags','')] + sorted([
                        (x.category,link(
                                        ','.join(sorted(x.detail.replace('+', '-').split(','))) or 'link',
                                        x.href,
                                        new_tab=x.new_tab,
                                        ))
                        for x in flags
                        ]
                ))
            
        return flag_html

# Usage:
# >>> from flagging.models import FlagSet,Flag
# >>> fs=FlagSet()
# >>> fs=FlagSet(ws_id=5,source='test')
# >>> fs.save()
# >>> f=Flag(wsa_id=5230,run=fs,category='Some problem',href='https://eastshore.com')
# >>> f.save()
# >>> f.detail='place with info'
# >>> f.save()
# >>> fs.enabled=False
# >>> fs.save()
