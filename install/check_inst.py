# Needs venv; crontab entry should run under correct py3

def main():
    import os
    pgm = os.path.basename(__file__)
    import datetime
    from dtk.text import fmt_time
    print(
        fmt_time(datetime.datetime.now()),
        pgm,
        'running',
        )
    from aws_op import Machine
    mch = Machine(name='')
    from io import StringIO
    from dtk.files import Quiet
    capture_out = StringIO()
    with Quiet(replacement=capture_out) as tmp:
        mch.do_inst_check()
    output = capture_out.getvalue()
    from dtk.alert import slack_send
    slack_send(f'on {os.uname()[1]} {pgm} report\n'+output)

if __name__ == "__main__":
    main()
