default:
	@echo "USAGE: make input|build|clean"

# The source data used to build this collection is no longer available.
# Copies of the pre-versioned outputs (with NCATS- prefix added to keys)
# are checked in as v1.

FILE_CLASS=ncats

include ../version_tools/setup.mk
include versions.mk

OFILE_COL=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
OFILE_DPI=$(FILE_CLASS).default.$(OUT_VER).evidence.tsv
OUTPUTS=$(OFILE_COL) $(OFILE_DPI)

input:
	false # not supported

build: $(OUTPUTS)

