
class FLock:
    '''Blocking, advisory, cross-process lock
    '''
    def __init__(self, filename):
        self.filename = filename
        # This will create it if it does not exist already
        self.handle = open(filename, 'w')
        # We will also hold a threading lock any time we hold a file lock,
        # to prevent same-process multiple access as well.
        from threading import Lock
        self.lock = Lock()

    def acquire(self):
        self.lock.acquire()
        import fcntl
        fcntl.flock(self.handle, fcntl.LOCK_EX)
    def release(self):
        import fcntl
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.lock.release()
    def __del__(self):
        # If opening the lock handle in init failed, handle won't be set,
        # and there's nothing to close; just ignore this case, you will have
        # already gotten the exception at startup.
        if hasattr(self, 'handle'):
            self.handle.close()
    # allow FLock to work as a context manager
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self,type,value,traceback):
        self.release()
