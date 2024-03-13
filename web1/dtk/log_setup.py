
"""
Scripts can use this module to setup apply a default logging setup when
run outside of the django environment (which already does so).

In particular, it will print nicely formatted (>=INFO) logs to the console.

It also provides some default argparse args for controlling verbose/quiet.
"""

import logging
logger = logging.getLogger(__name__)

def setupLogging(args=None):
    import logging.config

    # Setup the default configuration we use with django.
    from web1.settings import LOGGING
    logging.config.dictConfig(LOGGING)
    
    import sys
    # Disable buffering so that we get immediate and in-order output.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    if args:
        # Apply our overrides via 'incremental'.
        # Note that incremental mostly just limits you to setting levels.
        cfg = dict(LOGGING)
        if args.quiet:
            cfg['handlers']['console']['level'] = 'WARNING'
        elif args.verbose:
            cfg['handlers']['console']['level'] = 'DEBUG'
        cfg['incremental'] = True
        logging.config.dictConfig(LOGGING)

    # Disable logging to django.log unless specifically requested.
    if not args or not args.log_to_django:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if 'SysLogHandler' in str(handler):
                root_logger.removeHandler(handler)


    import sys
    logger.info("Logging started for %s", sys.argv)


def addLoggingArgs(parser):
    group = parser.add_argument_group("Logging")
    group.add_argument('-q', '--quiet', action='store_true', help='Display only WARN or above')
    group.add_argument('-v', '--verbose', action='store_true', help='Display DEBUG or above')
    group.add_argument('--log-to-django', default=False, action='store_true', help='Log to our common django.log file')
