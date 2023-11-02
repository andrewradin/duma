default:
	@echo "USAGE: make input|build|publish_ws|publish_s3|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=aact

WS=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

# AACT info:
# look https://aact.ctti-clinicaltrials.org/pipe_files Monthly Archive, for the latest version
# https://aact.ctti-clinicaltrials.org/schema
# https://aact.ctti-clinicaltrials.org/data_dictionary

# updated approximately monthly
ZIPFILE=$(AACT_VER)_pipe-delimited-export.zip
URL=https://aact.ctti-clinicaltrials.org/static/exported_files/monthly
ZIP_DNLD=$(DNLD)$(ZIPFILE)

# Notes on extracting toxicity data:
# All adverse events for a study are recorded in the 'reported_events' table.
# In addition to the NCT_ID, these are also keyed by 'result_group_id'. The
# result groups have type fields that would need to be interpreted for proper
# filtering. The reported events records seem to be roll-ups of similar
# events (they have an event_count field, e.g.) and various categorizations
# (event_type, organ_system, adverse_event_term, etc.). More investigation
# is required to fully understand the interpretation of these fields.

INFILES=\
	browse_conditions.txt \
	interventions.txt \
	studies.txt \
	# end of list
KOL_FILES=\
	facility_contacts.txt \
	facility_investigators.txt \
	overall_officials.txt \
	responsible_parties.txt \
	facilities.txt \
	# end of list
SPONSORS_IN=sponsors.txt
INS_FROM_ZIP=$(INFILES) $(KOL_FILES) $(SPONSORS_IN)
DRUGNAMES_FILE=drugnames.v$(DRUGNAMES_VER).tsv
DRUGNAMES_PATH=$(WS)/drugnames/$(DRUGNAMES_FILE)

TMP=tmp.out

OFILE_STEM=$(FILE_CLASS).$(OUT_VER)
OUT_TREAT=$(OFILE_STEM).disease_drug_matrix.tsv
OUT_KOL=$(OFILE_STEM).study_contacts.tsv
OUT_SPONSORS=$(OFILE_STEM).sponsors.tsv

OUT_SQL=\
    $(OFILE_STEM).studies.sqlsv \
    $(OFILE_STEM).drugs.sqlsv \
    $(OFILE_STEM).diseases.sqlsv \
    # end of list
    

OUTPUTS=$(OUT_TREAT) $(OUT_KOL) $(OUT_SPONSORS) $(OUT_SQL)

input: $(INFILES) $(KOL_FILES) $(SPONSORS_IN) $(DRUGNAMES_PATH)

build: $(OUTPUTS)

clean:
	rm -f *.txt *.docx *.xlsx tmp*

$(OUT_TREAT) $(OUT_SQL): $(INFILES) $(DRUGNAMES_PATH)
	./get_experimental_drugs.py $(DRUGNAMES_PATH) --output-prefix $(OFILE_STEM) > $(TMP)
	mv $(TMP) $(OUT_TREAT)

$(OUT_KOL): $(KOL_FILES)
	./get_contacts.py $(KOL_FILES) > $(TMP)
	mv $(TMP) $@

$(OUT_SPONSORS): $(SPONSORS_IN)
	./parse_sponsors.sh $(SPONSORS_IN) > $(TMP)
	mv $(TMP) $@

$(INS_FROM_ZIP): $(ZIP_DNLD)
	unzip -o $(ZIP_DNLD) $(INS_FROM_ZIP)
	touch $(INS_FROM_ZIP)

$(DRUGNAMES_PATH):
	$(S3_TOOL) drugnames $(DRUGNAMES_FILE)

show_latest_version:
	@echo
	@echo "Go to https://aact.ctti-clinicaltrials.org/pipe_files"
	@echo " to identify the latest daily archive, e.g. 20220830"
	@echo "Use that value to update AACT_VER"
	@echo
	@echo "Starting Sep2022 the pipe files need to be manually downloaded,"
	@echo " renamed e.g., $(ZIPFILE)"
	@echo " and copied over to the download directory"
	@echo
	@echo "Also update to the latest DRUGNAMES_VER"
	@echo
	@echo "After that follow the normal input, build, publish_s3"
