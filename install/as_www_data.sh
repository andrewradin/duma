#!/bin/bash

set -e

USER=www-data
sudo -H -u $USER $*
