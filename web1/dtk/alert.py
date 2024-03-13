import logging
logger = logging.getLogger(__name__)


def slack_send(msg, add_host=False):
    if add_host:
        import os
        host = os.uname()[1]
        msg = f'(host:{host}) {msg}'
    assert '"' not in msg,'double quote not supported in message'
    logger.info(f"Sending '{msg}' to slack")
    import os
    if 'PYTEST_CURRENT_TEST' in os.environ:
        # Don't spam slack from tests.
        logger.warning(f"Skipping slack message '{msg}' in unit test")
        return
    from path_helper import PathHelper
    import subprocess
    pgm = PathHelper.website_root + 'scripts/slack_send.sh'
    subprocess.check_call([
            pgm,
            msg,
            ])
