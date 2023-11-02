default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

# This directory produces a list of drugnames that AACT uses to locate
# drug names within free-form text. This code was factored out of AACT
# during the conversion to versioned ETL to provide some dependency
# isolation between AACT and individual collections this data is sourced
# from.
#
# v1 is an ad-hoc set reverse-engineered from the AACT output to produce
# the legacy AACT files.
FILE_CLASS=drugnames

# Note that currently, we extract names from all collections used for
# matching. If we want an more restricted subset, change the following
# to be like the similar code block in name2cas.
FIND=../matching/find_etl_file.py
INGREDIENTS_FILE=matching.full.$(MATCHING_VER).ingredients.tsv
INGREDIENTS_PATH=$(shell $(FIND) $(INGREDIENTS_FILE))
COLL_FILES=$(shell cat $(INGREDIENTS_PATH))
COLL_PATHS=$(shell $(FIND) $(COLL_FILES))

TMP=tempfile.tmp
EXTRA_INPUTS=\
	$(WS_DIR)drugsets/create.ttd.full.tsv \
	# end of list

OFILE=$(FILE_CLASS).$(OUT_VER).tsv
OUTPUTS=$(OFILE)

show_latest_versions:
	@echo 'This follows the normal process, other than it has no "make input" step'

build: $(OUTPUTS)

clean:
	-rm *.tsv
	-rm *.json.gz
	-rm $(TMP) $(TMP2)

$(OFILE): $(COLL_PATHS) $(EXTRA_INPUTS)
	./get_collection_drugnames.py $(COLL_PATHS) $(EXTRA_INPUTS) > $(TMP)
	sort -o $(TMP) $(TMP)
	mv $(TMP) $@
