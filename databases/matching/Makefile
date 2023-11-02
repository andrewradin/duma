default:
	@echo "USAGE: make input|build|publish_s3|clean"

FILE_CLASS=matching

include ../version_tools/setup.mk
include versions.mk

# This directory is responsible for reconciling and combining different
# compound libraries so they can be used by Duma. The main flow is:
# - examine attribute files from all compound libraries to identify
#   molecules duplicated across (and sometimes within) libraries;
#   for each 'cluster' of duplicate molecules, a single key is chosen
#   to represent that molecule (the 'dpimerge_id'); molecules
#   without duplicates use their native collection key as a dpimerge_id
#   (Note that this requires that the key namespaces of different
#   collections don't overlap. This is usually true since most collections
#   begin every key with a unique collection-specific prefix. Where this
#   isn't true, we need to add such a prefix when the collection is
#   extracted.)
# - DPI information from different collections is combined and re-keyed
#   with the dpimerge_id for the compound; multiple combinations can
#   be produced

######
# XXX validate these tools
# compare_ws_stats - report on changes between the local finalized files
#    and the copies in 2xar/ws

ATTR_CHECK_FILE=check.$(OUT_VER)
CLUSTER_DEF_FILE=$(FILE_CLASS).full.$(OUT_VER).clusters.tsv
PROP_FILE=$(FILE_CLASS).full.$(OUT_VER).props.tsv
INGREDIENTS_FILE=$(FILE_CLASS).full.$(OUT_VER).ingredients.tsv
DPIMERGE_OUTPUTS=$(subst VERSION,$(OUT_VER),$(DPIMERGE_OUT_STEMS))
DPIMERGE_ARGS=$(subst VERSION,$(OUT_VER),$(DPIMERGE_ARG_STEMS))
COMBO_OUTPUTS=$(subst VERSION,$(OUT_VER),$(COMBO_OUT_STEMS))
COMBO_INPUTS=$(subst VERSION,$(OUT_VER),$(COMBO_IN_STEMS))
MOA_OUTPUTS=$(subst VERSION,$(OUT_VER),$(MOA_OUT_STEMS))

# Use a flag file to track whether this step was completed.
PARTIALS_DONE=$(CLUSTER_DEF_FILE).partials_done

# This isn't included as one of the outputs.  Instead it is
# listed as an output by the moa directory and uploaded there.
MOA_ATTRS=moa_attrs.$(OUT_VER).tsv

WS=$(shell ../../web1/path_helper.py storage)
UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
ENSEMBL_UNIPROT_CONVERTER=$(WS)/$(UNI_BUCKET)/$(UNI_FILE)


OTHER_OUTPUTS=$(PROP_FILE) $(INGREDIENTS_FILE) \
	$(DPIMERGE_OUTPUTS) $(COMBO_OUTPUTS) $(MOA_OUTPUTS)
OUTPUTS=$(CLUSTER_DEF_FILE) $(OTHER_OUTPUTS)

MATCH_INPUT_PATHS=$(shell ../matching/find_etl_file.py $(MATCH_INPUTS))

input: $(MATCH_INPUT_PATHS) $(ATTR_CHECK_FILE)

$(ATTR_CHECK_FILE): $(MATCH_INPUT_PATHS)
	@set -e ;\
	for FILE in $(MATCH_INPUT_PATHS) ;\
	do \
		echo checking $$FILE ;\
		./check_create_file.py $$FILE ;\
	done
	touch $@

$(ENSEMBL_UNIPROT_CONVERTER):
	 $(S3_TOOL) $(UNI_BUCKET) $(UNI_FILE)

build: $(ATTR_CHECK_FILE) $(CLUSTER_DEF_FILE) \
        $(PARTIALS_DONE) $(DPIMERGE_INPUTS) $(OTHER_OUTPUTS)

show_latest_version:
	@echo
	@echo 'To update versions.py run:'
	@echo './next_version_configuration.py'
	@echo 'Then copy that output to versions.py and update the description'
	@echo
	@echo 'After that run the normal process:'
	@echo make input
	@echo make build
	@echo make publish_s3

$(DPIMERGE_OUTPUTS): $(PARTIALS_DONE) $(CLUSTER_DEF_FILE) $(DPIMERGE_INPUTS)
	./dpi_merge.py --cluster $(CLUSTER_DEF_FILE) $(DPIMERGE_ARGS)

$(COMBO_OUTPUTS): $(COMBO_INPUTS)
	./dpi_merge.py --combos $(COMBO_OUTPUTS)

$(MOA_OUTPUTS) $(MOA_ATTRS): $(DPIMERGE_OUTPUTS) $(ENSEMBL_UNIPROT_CONVERTER)
	./dpi_transform.py \
		--rekey \
		-u $(ENSEMBL_UNIPROT_CONVERTER) \
		-v $(OUT_VER) \
		-o $(MOA_OUTPUTS) \
		-i $(DPIMERGE_OUTPUTS) \
		-a $(MOA_ATTRS) \
		--merge-prev


$(PROP_FILE): prop_dictionary.tsv.master
	cp $< $@

$(INGREDIENTS_FILE): versions.py
	echo $(MATCH_INPUTS) > $@

# this runs enough of the clustering algorithm to examine logs and
# output, without affecting make targets
test_matching:
	mkdir -p $(LOGDIR)
	./match_tool.py --trim --save-archive --write-cluster-dump \
            --falsehoods falsehoods.tsv \
			--logdir $(LOGDIR) $(MATCH_INPUT_PATHS) \
			> $(LOGDIR)/match.log

$(CLUSTER_DEF_FILE): $(MATCH_INPUT_PATHS)
	mkdir -p $(LOGDIR)
	./match_tool.py --trim --save-archive --write-cluster-dump \
            --falsehoods falsehoods.tsv \
			--logdir $(LOGDIR) $(MATCH_INPUT_PATHS) \
			> $(LOGDIR)/match.log
	cp $(LOGDIR)/cluster_dump.out $@

$(PARTIALS_DONE): $(CLUSTER_DEF_FILE)
	# After the clustering file is completed, any incomplete source
	# directories need to be re-built. This needs to happen before
	# the rest of the outputs are built, so that things like DPI files
	# are in place.
	./finish_partial_inputs.py $(MATCH_INPUTS)
	touch $(PARTIALS_DONE)

TS:=$(shell date +%Y.%m.%d-%H:%M:%S)
LOGDIR=log/$(TS)

