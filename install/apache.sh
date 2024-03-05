#!/bin/sh

# graceful will allow threads to finish before having them restart,
# allowing any running jobs to complete.
sudo service apache2 graceful

