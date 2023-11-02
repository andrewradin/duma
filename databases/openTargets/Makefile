default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=openTargets

PARSE_PRGM=./parse_opentargets_json.py

WS=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

BASE_DNLD_URL=ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform

# NOTE: Helpful hint if you're trying to debug these files... there are also JSONL versions online, which are easier to
# trawl through by hand.  All the field names should be the same.
OTARG_OVERALL_INPUT_DIR=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_VER).association_data
OTARG_DIS_INPUT_DIR=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_VER).disease_data
OTARG_BYTYPE_INPUT_DIR=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_VER).association_data_by_type

# Unfortunately you're going to have to manually check https://www.targetvalidation.org/downloads/data for the
# correct suffixes on these filenames as they are non-standardized to version.
OTARG_TARG_SAFETY_DNLD_URL=$(BASE_DNLD_URL)/$(OPENTARGETS_NONEFF_VER)/input/annotation-files/known_target_safety-$(TARG_SAFETY_SUFFIX).json
OTARG_TARG_SAFETY_INPUT=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_NONEFF_VER).known_target_safety.json

# This data looks very hard to use, going to ignore for now.
# OTARG_EXP_TOX_DNLD_URL=$(BASE_DNLD_URL)/$(OPENTARGETS_VER)/input/annotation-files/experimental-toxicity-2020-04-07.tsv
# OTARG_EXP_TOX_INPUT=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_VER).experimental_toxicity.tsv

OTARG_TARG_TRACT_DNLD_URL=$(BASE_DNLD_URL)/$(OPENTARGETS_NONEFF_VER)/input/annotation-files/tractability_buckets-$(TRACT_SUFFIX).tsv
OTARG_TARG_TRACT_INPUT=$(DNLD)$(FILE_CLASS).$(OPENTARGETS_NONEFF_VER).tractability.tsv



UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
ENSEMBL_UNIPROT_CONVERTER=$(WS)/$(UNI_BUCKET)/$(UNI_FILE)

OFILE_STEM=$(FILE_CLASS).$(OUT_VER)
OFILE=$(OFILE_STEM).data.tsv.gz
OFILE2=$(OFILE_STEM).names.tsv

OUT_TARG_SAFETY=$(OFILE_STEM).target_safety.tsv
OUT_TARG_TRACT=$(OFILE_STEM).tractability.sqlsv


TMP=tempfile.tmp

INPUTS=\
	$(OTARG_OVERALL_INPUT_DIR) \
	$(OTARG_DIS_INPUT_DIR) \
	$(OTARG_BYTYPE_INPUT_DIR) \
	$(ENSEMBL_UNIPROT_CONVERTER) \
	$(OTARG_TARG_SAFETY_INPUT) \
	$(OTARG_TARG_TRACT_INPUT) \
    # end of list

OUTPUTS=\
	$(OFILE) \
	$(OFILE2) \
	$(OUT_TARG_SAFETY) \
	$(OUT_TARG_TRACT) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '========================================='
	@echo 'Visit https://blog.opentargets.org/tag/release-notes/ to determine the latest release'
	@echo ' of the openTargets data and update OPENTARGETS_VER in versions.py accordingly'
	@echo
	@echo 'Also, see the note in versions.py about OPENTARGETS_NONEFF_VER'
	@echo
	@echo 'ALSO check http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/<OPENTARGETS_VER>/input/annotation-files/ for the latest supplemental filenames'
	@echo '(targ safety, tractability, etc.) and update corresponding variables in versions.py.'
	@echo ' Note the easiest way I found to do this is by copying the link to files and pasting elsewhere'
	@echo
	@echo 'Then run:'
	@echo make input
	@echo make build
	@echo make publish_s3


$(OTARG_OVERALL_INPUT_DIR):
	mkdir -p $(OTARG_OVERALL_INPUT_DIR).tmp
	rsync -rpltz --info=progress2 --delete \
		rsync.ebi.ac.uk::pub/databases/opentargets/platform/$(OPENTARGETS_VER)/output/etl/parquet/associationByOverallIndirect/ \
		$(OTARG_OVERALL_INPUT_DIR).tmp/
	mv $(OTARG_OVERALL_INPUT_DIR).tmp $(OTARG_OVERALL_INPUT_DIR)

$(OTARG_BYTYPE_INPUT_DIR):
	mkdir -p $(OTARG_BYTYPE_INPUT_DIR).tmp
	rsync -rpltz --info=progress2 --delete \
		rsync.ebi.ac.uk::pub/databases/opentargets/platform/$(OPENTARGETS_VER)/output/etl/parquet/associationByDatatypeIndirect/ \
		$(OTARG_BYTYPE_INPUT_DIR).tmp/
	mv $(OTARG_BYTYPE_INPUT_DIR).tmp $(OTARG_BYTYPE_INPUT_DIR)

$(OTARG_DIS_INPUT_DIR):
	mkdir -p $(OTARG_DIS_INPUT_DIR).tmp
	rsync -rpltz --info=progress2 --delete \
		rsync.ebi.ac.uk::pub/databases/opentargets/platform/$(OPENTARGETS_VER)/output/etl/parquet/diseases/ \
		$(OTARG_DIS_INPUT_DIR).tmp/
	mv $(OTARG_DIS_INPUT_DIR).tmp $(OTARG_DIS_INPUT_DIR)

$(OTARG_TARG_SAFETY_INPUT):
	curl $(OTARG_TARG_SAFETY_DNLD_URL) > $(TMP)
	mv $(TMP) $(OTARG_TARG_SAFETY_INPUT)

$(OTARG_TARG_TRACT_INPUT):
	curl $(OTARG_TARG_TRACT_DNLD_URL) > $(TMP)
	mv $(TMP) $(OTARG_TARG_TRACT_INPUT)

$(OUT_TARG_SAFETY): $(OTARG_TARG_SAFETY_INPUT) $(ENSEMBL_UNIPROT_CONVERTER)
	./parse_target_safety.py -i $< -o $@ -u $(ENSEMBL_UNIPROT_CONVERTER)

$(OUT_TARG_TRACT): $(OTARG_TARG_TRACT_INPUT) $(ENSEMBL_UNIPROT_CONVERTER)
	./parse_tractability.py -i $< -o $@ -u $(ENSEMBL_UNIPROT_CONVERTER)

$(ENSEMBL_UNIPROT_CONVERTER):
	 $(S3_TOOL) $(UNI_BUCKET) $(UNI_FILE)

$(OFILE) $(OFILE2): $(OTARG_INPUT_DIR) $(OTARG_DIS_INPUT_DIR) $(OTARG_BYTYPE_INPUT_DIR) $(ENSEMBL_UNIPROT_CONVERTER)
	$(PARSE_PRGM) \
		--otarg-disease-dir $(OTARG_DIS_INPUT_DIR) \
		--otarg-overall-dir $(OTARG_OVERALL_INPUT_DIR) \
		--otarg-bytype-dir $(OTARG_BYTYPE_INPUT_DIR) \
		-u $(ENSEMBL_UNIPROT_CONVERTER)
	gzip temp_data.tsv
	mv temp_data.tsv.gz $(OFILE)
	mv temp_names.tsv $(OFILE2)

clean:
	rm -rf *.tsv *.jon.gz $(TMP)
