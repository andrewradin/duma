from path_helper import PathHelper

class LogRepoInfo:
    def __init__(self,job_id,**kwargs):
        job_id = int(job_id)
        self._path_parts = [
                PathHelper.lts,
                ('log%d'%(job_id%10)), # repo name
                ('%03d/%d' % (job_id % 1000,job_id)), # sharded job dir
                'publish/bg_log.txt', # viewable file name
                ]
    def get_repo(self):
        from dtk.lts import LtsRepo
        return LtsRepo.get(
                self._path_parts[1],
                PathHelper.cfg('lts_branch'),
                )
    def log_path(self):
        import os
        return os.path.join(*self._path_parts)
    def progress_path(self):
        import os
        return os.path.join(*self._path_parts[:-1]+['progress'])
    def fetch_log(self):
        import os
        log_path = self.log_path()
        if not os.path.exists(log_path):
            self.get_repo().lts_fetch(self._path_parts[2])
    def push_log(self):
        self.get_repo().lts_push(self._path_parts[2])

