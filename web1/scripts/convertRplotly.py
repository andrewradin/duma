#!/usr/bin/env python3

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='convert R index.html to PlotlyPlot save format'
            )
    parser.add_argument('R_file')
    parser.add_argument('plotly_file')
    args=parser.parse_args()

    from dtk.plot import PlotlyPlot
    pp = PlotlyPlot.build_from_R(args.R_file)
    pp.save(args.plotly_file)

