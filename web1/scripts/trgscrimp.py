#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys

import os
from six.moves import cPickle as pickle
import logging
logger = logging.getLogger(__name__)


method=None
from dtk.target_importance import Cache

# Note that each process will have its own copy of this cache, which
# does reduce the usefulness of it.
# Could possibly see some benefits from prepoulating it.
# But it still helps out within each process as-is.
cache = Cache()

def run_piece(x):
    try:
        from dtk.target_importance import TargetScoreImportance
        return TargetScoreImportance.run_piece(x, method, cache)
    except Exception as e:
        logger.info(("Failed running piece: ", x.get('description', 'unknown')))
        import traceback
        traceback.print_exc()
        return {}

def run(input_file, outdir, num_cores):
    logger.info("Loading input pickle")
    with open(input_file, 'rb') as f:
        input_data = pickle.loads(f.read())

    global method
    method = input_data['method']

    logger.info("Computing %d scores" % len(input_data['pieces']))

    piece_data = input_data['pieces']

    # Sort pieces by increasing order of duration
    # We also keep track of their original index, so we can reorder at the end
    def order_fn(x):
        if x[1] is None:
            return 0
        else:
            return -x[1]['predicted_duration']
    ordered_piece_data = sorted(zip(list(range(len(piece_data))), piece_data), key=order_fn)
    in_data = [x[1] for x in ordered_piece_data]

    # TODO: Probably remove this eventually, mostly for memory, though
    # it probably doesn't hurt perf much
    #  (# physical cores is half virtual cores).
    import multiprocessing
    num_cores = min(num_cores, multiprocessing.cpu_count() // 2 + 1)


    from dtk.parallel import pmap

    # TODO: maxtasksperchild will cause the worker processes to get reset
    # after completing N tasks; this helps prevent memory from growing out of
    # control, at the cost of some runtime.
    # Will fix properly later by reducing memory usage and overcaching.
    # Task runtime is extremely uneven, most take less than a second, whereas some take upwards of 30 minutes.
    # We use a chunksize of 1, to avoid the chance of getting multiple very slow tasks in the same chunk.
    # This is especially important since we're sorting by size, which puts
    # all of the longer tasks near to each other in the data.
    unordered_output_data = pmap(run_piece, in_data, num_cores=num_cores, maxtasksperchild=200, chunksize=1, progress=True)

    ordered_output_data = zip(unordered_output_data, ordered_piece_data)
    # Sort by the original index, to put things back to their original order.
    ordered_output_data = sorted(ordered_output_data, key=lambda x: x[1][0])
    # Now that we're in order, drop the index.
    output_data = [x[0] for x in ordered_output_data]

    logger.info("Generating output pickle")
    output_file = os.path.join(outdir, 'output.pickle.gz')
    import gzip
    with gzip.open(output_file, 'wb') as f:
        f.write(pickle.dumps(output_data))



if __name__ == "__main__":
    import time

    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    parser = argparse.ArgumentParser(description='Run Target Score Importance')
    parser.add_argument("input", help="Input Pickle")
    parser.add_argument("outdir", help="Where to write output")
    parser.add_argument("cores", type=int, help="Number of cores to use")
    from dtk.log_setup import addLoggingArgs, setupLogging
    addLoggingArgs(parser)
    args = parser.parse_args()
    setupLogging(args)
    ts = time.time()
    logger.info("Running with %d cores" % args.cores)

    logger.info("Running target score importance")
    run(args.input, args.outdir, args.cores)

    logger.info("Took: " + str(time.time() - ts) + ' seconds')
