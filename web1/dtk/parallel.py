
import logging
logger = logging.getLogger(__name__)
g_static_args = None
g_raw_func = None
def init_fn(raw_func, static_args):
    global g_static_args
    global g_raw_func
    g_raw_func = raw_func
    g_static_args = static_args or {}

def func_wrapper(args):
    return g_raw_func(*args, **g_static_args)

force_fake_mp=False
def pmap(func, *args, static_args=None, num_cores=None, fake_mp=False, maxtasksperchild=None, progress=False, chunksize=1):
    """
    Convenience wrapper around a multiprocessing map with (potentially large)
    static initial arguments that are only passed to each worker once.

    *args is a list of iterables, matching the standard python3 map builtin.
    static_args is a dict that will be passed into each function call as kwargs.

    def find_path(start, end, graph):
       ...

    pmap(find_path, (0,1,2,3), (9,8,7,6), static_args={'graph': graph})
    """
    import multiprocessing
    if multiprocessing.current_process().daemon or force_fake_mp:
        # If we try to use real Pool here, it will fail.
        # We could use a threadpool instead, though, if we wanted to be closer to real.
        logger.debug("Using fake_mp in pmap due to daemon process")
        fake_mp = True

    assert len(args) > 0, "No inputs provided to map"

    map_args = zip(*args)

    if fake_mp:
        static_args = static_args or {}
        def fake_wrapper(args):
            return func(*args, **static_args)
        yield from map(fake_wrapper, map_args)
    else:
        static_args = (func, static_args)
        # NOTE: There is a much simpler implementation of this prior to sprint 258.
        # Unfortunately, it was very easy to run into rare deadlocks due to
        # a python 'bug' around mixing forks and threading. Doing almost anything
        # in the main thread (e.g. print statements) while running a map could result
        # in a deadlock. see https://bugs.python.org/issue6721
        #
        # print (and many other things) grabs a lock; if we happen to fork off a new
        # process while we're holding that lock, that process will hang the moment it tries to print,
        # regardless of whether the main process subsequently releases it.
        #
        # The 'proper' solution is to stop using fork - Pool supports both spawn and forkserver as
        # alternatives.  Unfortunately, these are much, much slower for initial static data transfer,
        # which is effectively free with fork.
        #
        # As a workaround, we're going to double fork.  This makes it much harder
        # to run into problems (you can still break things by explicitly using threads), at
        # the cost of some overhead as both inputs and outputs have an extra hop of serialization.
        # We fork once (via Process) at the start, when everything should be in a safe state.
        # Worker processes then fork off that initial fork, which leaves the main process free
        # to do things that take locks without worrying about a worker forking off it at a bad time.

        from multiprocessing import Pool, Queue, Process
        init_fn(*static_args)
        # Want these already resolved if you passed in a generator.
        map_args = list(map_args)

        q = Queue()
        def run():
            try:
                with Pool(
                        processes=num_cores,
                        maxtasksperchild=maxtasksperchild,
                        #initializer=init_fn,
                        #initargs=static_args,
                        ) as p:
                    for out in p.imap(func_wrapper, map_args, chunksize=chunksize):
                        q.put((out, None))
                    
                    # These help pytest-cov get proper data.
                    # See https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
                    p.close()
                    p.join()
            except Exception as e:
                # Normally we lose the traceback when we send the exception across
                # processes it seems, so print it here.
                import traceback as tb
                tb.print_exc()
                q.put((None, e))

        proc = Process(target=run)
        proc.start()
        if progress:
            desc = progress if isinstance(progress, str) else None
            import tqdm
            itr = tqdm.trange(len(map_args), desc=desc, mininterval=1, delay=1)
        else:
            itr = range(len(map_args))
        for _ in itr:
            out, err = q.get()
            if err:
                raise err
            else:
                yield out
        proc.join()


def chunker(lst, chunk_size=None, num_chunks=None, progress=False):
    """Divides a sequence into either {num_chunks} chunks of size {chunk_size}.

    Only one of chunk_size or num_chunks should be specified, the other is
    inferred.
    """
    assert chunk_size is None or num_chunks is None

    if num_chunks is None:
        num_chunks = (len(lst) + chunk_size - 1) // chunk_size

    if num_chunks == 0:
        return []

    # Even if you provided chunk_size, we might re-size it here to give more
    # even chunks (rather than having a tiny final chunk)
    chunk_size = (len(lst)) / num_chunks

    itr = range(num_chunks)
    if progress:
        from tqdm import tqdm
        itr = tqdm(itr, total=num_chunks)


    for i in itr:
        if i+1 == num_chunks:
            # Special case just to be safe, in case of rounding errors.
            yield lst[int(i*chunk_size):]
        else:
            yield lst[int(i*chunk_size):int((i+1)*chunk_size)]
