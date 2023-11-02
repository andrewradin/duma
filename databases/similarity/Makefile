FILE_CLASS=similarity

include ../version_tools/setup.mk
include versions.mk

FIND=../matching/find_etl_file.py
SEL=../matching/select_collections.py
INGREDIENTS_FILE=matching.full.$(MATCHING_VER).ingredients.tsv
INGREDIENTS_PATH=$(shell $(FIND) $(INGREDIENTS_FILE))
COLL_FILES=$(shell $(SEL) $(INGREDIENTS_PATH))
COLL_PATHS=$(shell $(FIND) $(COLL_FILES))


OUT_PREFIX=$(FILE_CLASS).default.$(OUT_VER)
OUT_STRUCT=$(OUT_PREFIX).fpdb.h5
OUT_STRUCT_META=$(OUT_PREFIX).struct_metadata.json

OUTPUTS=\
	$(OUT_STRUCT) \
	$(OUT_STRUCT_META) \
	# end of list

input: $(COLL_PATHS)

build: input $(OUTPUTS)

clean:
	-rm $(OUTPUTS) *.log

$(OUTPUTS): $(COLL_PATHS)
	./build_struct_db.py \
		-o $(OUT_PREFIX) \
		$(COLL_PATHS)