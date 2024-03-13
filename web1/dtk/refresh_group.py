class RefreshGroup:
    def __init__(self,json_file):
        import json
        self.ws_map={
                d['ws_id']:{'wf_job':d['wf_job']}
                for d in json.load(open(json_file))
                }
        # find scoreset for each workflow job
        from browse.models import ScoreSet
        for d in self.ws_map.values():
            d['ss'] = ScoreSet.objects.get(wf_job=d['wf_job'])
    def resume_info(self):
        return {
                k:v['ss'].id
                for k,v in self.ws_map.items()
                }

