# Needs venv; crontab entry should run under correct py3

def main():
    from path_helper import PathHelper
    script = PathHelper.website_root+'scripts/apikeys.py'
    import subprocess
    import sys
    subprocess.run([
            sys.executable, # pass conda python executable explicitly
            script,
            '--renew',
            ],
            check=True
            )


if __name__ == "__main__":
    main()
