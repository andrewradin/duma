from __future__ import print_function
import logging
logger = logging.getLogger(__name__)

def get_target_wsa_ids(ws,job_id,score,start,count,condensed=False):
    if count == 0:
        return []

    from runner.process_info import JobInfo
    bji=JobInfo.get_bound(ws,job_id)
    cat = bji.get_data_catalog()
    ordering = cat.get_ordering(score,True)

    if condensed:
        from dtk.enrichment import EMInput
        emi = EMInput(ordering, set())
        cond_emi = emi.get_condensed_emi()
        cond_scores = cond_emi.get_unlabeled_score_vector()
        ind = min(count, len(cond_scores)-1)
        thresh = cond_scores[ind]
        out = [x[0] for x in ordering if x[1] >= thresh]
        logger.info(f'Pulling {count} condensed wsas, total of {len(out)}, threshold score of {thresh:.2f}')
        return out
    else:
        return [x[0] for x  in ordering[start:start+count]]

def show_list_progress(l):
    prev_message = None
    for i,item in enumerate(l):
        message = 'progress: {:5.0f}%'.format(
                100*((i+1)/float(len(l)))
                )
        if message != prev_message:
            prev_message = message
            print(message)
        yield item

class FlaggerBase(object):
    # Note the absence of the ** in front of kwargs. This lets the base and
    # derived classes share a single dict for keyword parsing.  The derived
    # class should look like:
    #   def __init__(self,**kwargs):
    #       super(DerivedFlagger,self).__init__(kwargs)
    #       # pop my custom args from kwargs here
    #       assert not kwargs
    def __init__(self,kwargs):
        self._parms = dict(kwargs)
        self.ws_id = kwargs.pop('ws_id')
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self.job_id = kwargs.pop('job_id')
        self.score = kwargs.pop('score')
        self.start = kwargs.pop('start')
        self.count = kwargs.pop('count')
        self.condensed = kwargs.pop('condensed', False)
        self.wsa_qs = None
    def get_target_wsa_ids(self):
        return get_target_wsa_ids(
                self.ws,
                self.job_id,
                self.score,
                self.start,
                self.count,
                self.condensed,
                )
    def each_target_wsa(self,show_progress=True):
        if self.wsa_qs is None:
            from browse.models import WsAnnotation
            qs = WsAnnotation.objects.filter(pk__in=self.get_target_wsa_ids())
            self.wsa_qs = WsAnnotation.prefetch_agent_attributes(qs)
        if show_progress:
            return show_list_progress(list(self.wsa_qs))
        return self.wsa_qs
    def create_flag_set(self,source):
        from .models import FlagSet
        import json
        self.fs=FlagSet(
                ws_id=self.ws_id,
                source=source,
                settings=json.dumps(self._parms),
                )
        self.fs.save()
        return self.fs
    def create_flag(self,wsa_id,category,detail,href,fs=None):
        fs = fs if fs else self.fs
        from .models import Flag
        if len(detail) >= 255:
            # There's a max length, just truncate, flag text is probably
            # not super helpful if it's that long anyway.
            detail = detail[:250] + '...'
        f=Flag(
                wsa_id=wsa_id,
                run=fs,
                category=category,
                detail=detail,
                href=href,
                )
        f.save()

