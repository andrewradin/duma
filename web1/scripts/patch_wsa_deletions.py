# This was a one-shot script to fix a crash in workspace 43 caused by
# deleting wsa records. When I ran it, though, it ended up removing all
# the data from the output file, so I just removed the file instead.
# I checked this in for historical interest, in case we need something
# similar someday.
from browse.models import Workspace
import os
from path_helper import PathHelper

class DeletedWsaCleaner:
    def __init__(self,ws_id=43):
        self.ws = Workspace.objects.get(pk=ws_id)
        self.name_map = self.ws.get_wsa2name_map()
    def clean(self):
        # it turns out there's only one successful struct job in
        # ws 43, so I'm not writing the iteration code
        path=os.path.join(
                PathHelper.storage,
                str(self.ws.id),
                'struct',
                '12616',
                'output/similarities.csv'
                )
        self.clean_similarities_file(path)
    def clean_similarities_file(self,fn):
        from dtk.files import get_file_records,FileDestination
        if not os.path.exists(fn):
            return
        src = get_file_records(fn)
        rm_idxs = None
        for rec in src:
            if rm_idxs is None:
                # process header
                rm_idxs = []
                prefix = 'like_'
                for i,label in enumerate(rec):
                    if i == 0:
                        continue
                    assert label.startswith(prefix)
                    wsa_id = int(label[len(prefix):])
                    if wsa_id not in self.name_map:
                        rm_idxs.append(i)
                if not rm_idxs:
                    return # no columns need stripping
                # make sure deleting a column doesn't affect the indexes
                # of columns not yet deleted
                rm_idxs.reverse()
                # modify header
                for i in rm_idxs:
                    del(rec[i])
                # create output file
                opath = fn+'_tmp.csv'
                fd = FileDestination(opath,header=rec,delim=',')
                continue
            for i in rm_idxs:
                del(rec[i])
            fd.append(rec)
        # XXX move tmpfile over original
        # XXX doing this manually for now, as there's only one file to patch

