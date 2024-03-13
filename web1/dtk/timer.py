import datetime

class Timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self._start = datetime.datetime.now()
    def get(self):
        return datetime.datetime.now() - self._start
    def lap(self):
        end = datetime.datetime.now()
        delta = end - self._start
        self._start = end
        return delta

class BucketTimer:
    def __init__(self):
        self.reset()
    def reset(self):
        from collections import defaultdict
        self._accum = defaultdict(float)
        self._bucket = None
        self._start = None
    def start(self,bucket):
        self.stop()
        self._start = datetime.datetime.now()
        self._bucket = bucket
    def stop(self):
        if self._bucket:
            self._accum[self._bucket] += self._elapsed()
        self._bucket = None
    def _elapsed(self):
        if self._bucket is None:
            return 0
        return (datetime.datetime.now() - self._start).total_seconds()
    def get(self):
        l = list(self._accum.items())
        l.sort(key=lambda x:x[1],reverse=True)
        return l

