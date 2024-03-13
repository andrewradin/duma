from __future__ import print_function
class Relabeler:
    def __init__(self,ws_id,collections):
        from browse.models import WsAnnotation
        self.agent2wsa = {
                wsa.agent_id:wsa.id
                for wsa in WsAnnotation.objects.filter(ws_id=ws_id)
                }
        self.drug2agent = {}
        from drugs.models import Tag
        for name in collections:
            self.drug2agent.update({
                    (name,tag.value):tag.drug_id
                    for tag in Tag.objects.filter(
                            drug__collection__name=name+'.full',
                            prop__name=name+'_id',
                            )
                    })
    def read(self,path):
        from dtk.files import get_file_records
        for collection,key,score in get_file_records(path):
            try:
                agent = self.drug2agent[(collection,key)]
                wsa_id = self.agent2wsa[agent]
            except KeyError:
                print('skipping invalid key',key)
                continue
            score = float(score)
            yield (wsa_id,score)

