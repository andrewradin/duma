# Make sure the database reflects the correct resources when resizing
# an instance. This is installed as a @reboot crontab entry.
# This must be run under the virtualenv. The path can't be hardcoded in
# a shebang (it's not always /home/ubuntu) so the python executable is
# specified directly in the crontab.

from reserve import ResourceManager,default_totals
rm = ResourceManager()
rm.set_totals(default_totals())
