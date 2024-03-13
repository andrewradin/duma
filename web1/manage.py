#!/usr/bin/env python3
import os
import sys

import six
assert six.PY3, "This should be running with python3"

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
