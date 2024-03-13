#!/usr/bin/env python3

# on platform, this should be run like:
# sudo -u www-data migrate_ML_output.py

from __future__ import print_function
import sys
import os
import re
import shutil

# Make sure we can find PathHelper
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+'/..')

from path_helper import PathHelper,make_directory

class Migrator:
    pofa_re = re.compile(r'([0-9]+)_PofA.csv$')
    job_re = re.compile(r'[0-9]+$')
    def __init__(self):
        pass
    def run(self):
        done_dir = PathHelper.publish + 'ML_migrate_done'
        make_directory(done_dir)
        for ws in os.listdir(PathHelper.MLpublish):
            try:
                ws_id = int(ws)
                print('processing',ws)
                ws_root = os.path.join(PathHelper.MLpublish,ws)
                hist = os.path.join(ws_root,'history')
                self.scan_old_history(ws,hist)
                os.rename(ws_root,os.path.join(done_dir,ws))
            except Exception as ex:
                print(ws,'got exception',ex)
    def scan_old_history(self,ws,hist):
        for name in os.listdir(hist):
            m = self.pofa_re.match(name)
            if m:
                job_id = m.group(1)
                src = os.path.join(hist,name)
                dst_dir = os.path.join(
                                PathHelper.storage,
                                ws,
                                'ml',
                                job_id,
                                'output'
                                )
                make_directory(dst_dir)
                dst = os.path.join(dst_dir,'allPofA.csv')
                #print 'copying',src,'to',dst
                shutil.copyfile(src,dst)
                continue
            m = self.job_re.match(name)
            if m:
                job_id = name
                src_dir = os.path.join(hist,name,'full_vector')
                dst_dir = os.path.join(
                                PathHelper.publish,
                                ws,
                                'ml',
                                job_id,
                                )
                if os.path.exists(src_dir):
                    make_directory(dst_dir)
                    for name in os.listdir(src_dir):
                        if name == 'allProbabilityOfAssoications.csv':
                            continue
                        src = os.path.join(src_dir,name)
                        if os.path.isfile(src):
                            dst = os.path.join(dst_dir,name)
                            shutil.copyfile(src,dst)
                continue


if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(
            description="Move legacy ML output to CV-standard locations",
            )
    args = arguments.parse_args()

    m = Migrator()
    m.run()
