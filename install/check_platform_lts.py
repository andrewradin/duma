# Needs venv; crontab entry should run under correct py3

def main():
    # XXX Once we get some experience in production, we could expand
    # XXX processing to attempt automatic fixes to some kinds of LTS
    # XXX errors. Also, if the scans are taking too long, we can try
    # XXX parallelizing, or spliting the work between a fast scan for
    # XXX common errors, and do a slower, less-frequent scan for more
    # XXX rare conditions.
    import django_setup
    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        print(err)
        import sys
        sys.exit(1)
    try:
        Process.maint_request('lts_scan')
    except AssertionError:
        from dtk.alert import slack_send
        import os
        slack_send(
            f'on {os.uname()[1]} lts_scan request failed; already running?'
            )

if __name__ == "__main__":
    main()
