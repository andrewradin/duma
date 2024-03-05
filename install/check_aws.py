# Needs venv; crontab entry should run under correct py3

def main():
    errors = []
    from aws_op import Machine
    mch = Machine(name='dummy')
    from io import StringIO
    from dtk.files import Quiet
    for op in ('sg_check','s3_check','backup_check'):
        capture_out = StringIO()
        func = getattr(mch,'do_'+op)
        with Quiet(replacement=capture_out) as tmp:
            func()
        output = capture_out.getvalue()
        if output:
            errors.append((op,output))
    import os
    pgm = os.path.basename(__file__)
    import datetime
    from dtk.text import fmt_time
    print(
        fmt_time(datetime.datetime.now()),
        pgm,
        'errors:' if errors else 'ok',
        )
    for op,output in errors:
        print(op)
        print(output)
        print()
    if errors:
        from dtk.alert import slack_send
        slack_send(f'on {os.uname()[1]} {pgm} got errors on '+' '.join([
                x[0] for x in errors
                ]))

if __name__ == "__main__":
    main()
