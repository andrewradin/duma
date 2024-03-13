

import pytest
import logging
logger = logging.getLogger(__name__)

def make_mock_job(plugin_name, ws, scores):
    """Used for inserting job results into the db.
    
    Note that this only handles the basics of getting a datacatalog score.
    If there are additional output pieces to be inserted, that should be done
    manually.
    """
    from runner.process_info import JobInfo

    uji = JobInfo.get_unbound(plugin_name)
    settings = uji.settings_defaults(ws)['default']

    name = uji.get_jobnames(ws)[0]
    logger.info("Creating job with name %s", name)

    import json
    from runner.models import Process
    proc = Process.objects.create(
            name=name,
            status=Process.status_vals.SUCCEEDED,
            settings_json=json.dumps(settings),
            role=plugin_name,
            )

    bji = JobInfo.get_bound(ws, proc)

    logger.info("Writing scores to %s", bji.outfile)
    import os
    from path_helper import make_directory
    from dtk.files import FileDestination
    make_directory(os.path.dirname(bji.outfile))
    with FileDestination(bji.outfile) as f:
        for row in scores:
            f.append(row)

    assert ws.get_prev_job_choices(plugin_name) != 0

    logger.info("Creating fake bound job %s", bji)

