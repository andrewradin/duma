default: help

FILE_CLASS=name2cas

include ../version_tools/setup.mk
include versions.mk

FIND=../matching/find_etl_file.py
SEL=../matching/select_collections.py
INGREDIENTS_FILE=matching.full.$(MATCHING_VER).ingredients.tsv
INGREDIENTS_PATH=$(shell $(FIND) $(INGREDIENTS_FILE))
COLL_FILES=$(shell $(SEL) $(INGREDIENTS_PATH) $(SOURCE_COLLECTIONS))
COLL_PATHS=$(shell $(FIND) $(COLL_FILES))
OUTFILE=$(FILE_CLASS).$(OUT_VER).tsv
OUTPUTS=$(OUTFILE)

input: $(COLL_PATHS)

build: input $(OUTPUTS)

clean:
	-rm $(OUTPUTS) *.log

$(OUTFILE):
	./extract_cas.py $@ $(COLL_PATHS)
