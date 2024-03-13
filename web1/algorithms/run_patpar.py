#!/usr/bin/env python3

from runner.process_info import JobInfo

# This is a stub for a job that used to exist, but is no longer used.
# We keep a definition around just because entries for it still exist in the
# runner.Process table and the system doesn't like being unable to find it.
class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        super(MyJobInfo,self).__init__(ws,job,__file__,"Patent Parse","Patent Parse")
