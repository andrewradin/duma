#!/usr/bin/env python

import sys

class CrossMap:
    def __init__(self, chain_file):
        from path_helper import PathHelper
        from cmmodule.utils import read_chain_file
        (mapTree,targetChromSizes, sourceChromSizes) = read_chain_file(chain_file, print_table = False)
        self.mapping = mapTree
    # coords should be like
    # ['chr1', 246974830, 246974833, '+']
    def crossMap(self,coords):
        from cmmodule.utils import map_coordinates
        a = map_coordinates(self.mapping, *coords)
        # example of a: [('chr1', 246974830, 246974833,'+'), ('chr1', 248908207, 248908210,'+')]
        if not a:
            return None
        elif len(a) == 2:
            return a[1]
