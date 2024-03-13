#!/usr/bin/env python3

program_description='''\
Check integrity of LTS directories.
'''

from dtk.lazy_loader import LazyLoader
class LtsScan(LazyLoader):
    task = 'lts_scan'
    # set batch size to 1 to minimize latency of new job starts
    # (a large repo can take several minutes to process)
    batch_size = 1 # 10
    send_slack = True
    def scan_batch(self):
        from path_helper import PathHelper
        from dtk.lts import RepoChecker
        ok_results = set(['ok'])
        if PathHelper.cfg('machine_type') != 'platform':
            ok_results.add ('not replicated')
        errors = []
        from dtk.alert import slack_send
        for repo in self.batch_repos:
            # Note that we set remove to False here for both dev and
            # production systems. The fix code won't run unless the repo
            # is already on the current branch, so we don't need to worry
            # about pushing to dead branches. Since dev branches aren't
            # replicated, this will just do a local push, not an S3 write.
            # Since import_prod_db.sh runs a check with remove=True before
            # installing a new database, this script should never see an
            # unclean condition on an out-of-date branch anyway.
            rc = RepoChecker(
                    repo_path=PathHelper.lts+repo,
                    fix=True,
                    remove=False,
                    )
            result = rc.scan()
            aborting = False
            if result not in ok_results:
                fixed = any(result.endswith('; '+x) for x in ok_results)
                if fixed:
                    self.logger.warning("repo %s - %s",repo,result)
                    errors.append((repo,result))
                else:
                    aborting = True
                    self.logger.error("repo %s - %s",repo,result)
            for info in rc.unclean_report:
                self.logger.info("repo %s - %s",repo,info)
            if aborting:
                slack_send(
                    '; '.join(
                            'aborting lts_scan',
                            'unfixable repo',
                            'see django.log',
                            'scan must be restarted manually',
                            ),
                    add_host=True,
                    )
                raise RuntimeError('fix failed')
        if errors and self.send_slack:
            slack_send(
                'see django.log for fixes to LTS repo(s): '+', '.join(
                        x[0] for x in errors
                        ),
                add_host=True,
                )
    def complete_batch(self):
        from runner.models import Process
        next_start = self.start+self.batch_size
        total = len(self.all_repos)
        if next_start >= total:
            Process.maint_release(self.task)
        else:
            self.detail['start'] = next_start
            self.detail['progress'] = f"scanned {next_start} of {total}"
            Process.maint_yield(self.task,detail=self.detail)
    def _logger_loader(self):
        import logging
        return logging.getLogger(self.task)
    def _detail_loader(self):
        from runner.models import Process
        detail = Process.maint_detail(self.task)
        return detail
    def _all_repos_loader(self):
        from path_helper import PathHelper
        from dtk.files import scan_dir
        from dtk.data import cond_int_key
        return sorted(
                scan_dir(PathHelper.lts,output=lambda x:x.filename),
                key = lambda x: cond_int_key(x),
                )
    def _batch_repos_loader(self):
        end = self.start+self.batch_size
        return self.all_repos[self.start:end]
    def _start_loader(self):
        return self.detail.get('start',0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    args = parser.parse_args()

    import django_setup
    scanner = LtsScan()
    from path_helper import PathHelper
    scanner.send_slack = PathHelper.cfg('lts_scan_slack_report')
    scanner.scan_batch()
    scanner.complete_batch()
