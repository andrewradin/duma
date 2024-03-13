from __future__ import print_function

import vcr

def before_record_request(req):
    # We don't want to record anything s3 or IAM related or localhost reqs.
    # The s3 data will be big and the IAM data could have keys in it.
    # The localhost requests are usually from selenium.
    # Not clear why plotly needs remote data, but it's non-deterministic,
    # ignore for now.
    # Let's just skip anything with the word auth in it - if that becomes
    # too liberal, we can be more specific later.
    skip_patterns = ['/iam/', 's3.amazonaws.com', 'amazonaws.com',
            'http://localhost', 'http://127.0.0.1', 'https://api.plot.ly',
            '169.254.169.254', 'auth'
            ]

    for skip_pattern in skip_patterns:
        if skip_pattern in req.uri:
            # This is super spammy, particuarly s3 and selenium.
            #print("NOT recording/replaying %s" % req.uri)
            return None
    print("Recording/replaying %s" % req.uri)
    return req

std_vcr = vcr.VCR(
        before_record_request=before_record_request,
        match_on=['uri', 'method', 'body']
        )
