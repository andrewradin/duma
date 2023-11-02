#!/usr/bin/env python

def run(out_prefix, inputs):
    from dtk.metasim import StructSim
    StructSim.precompute_to_files(out_prefix, inputs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse OpenTargets tractability')
    parser.add_argument('-o', '--out-prefix', help = 'Where to write the output')
    parser.add_argument('inputs', nargs='+', help = 'attrs input files')
    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
