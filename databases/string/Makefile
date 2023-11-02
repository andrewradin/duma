default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=string
include ../version_tools/setup.mk
include versions.mk


PRCS_PRGM=./parse_string.py
WS_PPI=$(shell ../../web1/path_helper.py ppi)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_STRG=$(shell ../../web1/path_helper.py storage)
SOURCE=string

V=$(STRING_VER)
Vd=$(subst .,-,$(V))
URL=https://version-$(Vd).string-db.org/download/protein.links.v$(V)/
FILE_NAME=9606.protein.links.v$(V).txt.gz
INPUT=$(WS_DNLD)/$(FILE_NAME)

# USE_ACTIONS is NOT defined, so these and a similar conditional below are
# effectively commented out. References to these variables harmlessly expand
# to nothing wherever else they are used. This can be re-instated if String
# resumes publishing an actions file.
ifdef $(USE_ACTIONS)
ACTIONS_URL=https://version-$(Vd).string-db.org/download/protein.actions.v$(V)/
ACTIONS_FILE_NAME=9606.protein.actions.v$(V).txt.gz
ACTIONS_INPUT=$(WS_DNLD)/$(ACTIONS_FILE_NAME)
ACTIONS_PARMS=-a $(ACTIONS_INPUT)
endif

TMP=tempfile.tmp

UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Protein_Ensembl.tsv
UNI_BUCKET=uniprot
ENSEMBL_UNIPROT_CONVERTER=$(WS_STRG)/$(UNI_BUCKET)/$(UNI_FILE)

UNIPROT=uniprot_to_ensp.$(UNIPROT_VER).tsv

PPI_FILE=$(FILE_CLASS).default.$(OUT_VER).tsv
PPI_SQL_FILE=$(FILE_CLASS).default.$(OUT_VER).sqlsv

#$(info outver=${OUTVER} source=${SOURCE} ppi_file=${PPI_FILE})

OUTPUTS=\
	$(PPI_FILE) \
	$(PPI_SQL_FILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

show_latest_version:
	@echo
	@echo '=============================================='
	@echo 'Visit https://string-db.org/cgi/access.pl?footer_active_subpage=archive'
	@echo ' to identify the latest version of String'
	@echo ' update STRING_VER in versions.py accordingly'
	@echo 'Also update to the latest uniprot version there.'
	@echo 'Then run the normal process:'
	@echo 'make input'
	@echo 'make build'
	@echo 'make publish_s3'

input: $(INPUT) $(ACTIONS_INPUT) $(UNIPROT)
build: $(OUTPUTS)

$(INPUT):
	curl -f -L $(URL)$(FILE_NAME) > $(TMP)
	mv $(TMP) $@

ifdef $(USE_ACTIONS)
$(ACTIONS_INPUT):
	curl -f -L $(ACTIONS_URL)$(ACTIONS_FILE_NAME) > $(TMP)
	mv $(TMP) $@
endif

$(UNIPROT): $(ENSEMBL_UNIPROT_CONVERTER)
	grep 'ENSP' $(ENSEMBL_UNIPROT_CONVERTER) > $(UNIPROT)

$(ENSEMBL_UNIPROT_CONVERTER):
	 $(S3_TOOL) $(UNI_BUCKET) $(UNI_FILE)

$(PPI_FILE): $(INPUT) $(UNIPROT) $(ACTIONS_INPUT)
	zcat $(INPUT) > input.ssv
	$(PRCS_PRGM) -i input.ssv $(ACTIONS_PARMS) -u $(UNIPROT) -o temp.tsv
	../matching/standardize_pi.py temp.tsv > temp2
	mv temp2 $@
	rm temp.tsv input.ssv

$(PPI_SQL_FILE): $(PPI_FILE)
	../../web1/scripts/tsv_convert.py \
		-i $(PPI_FILE) -o $(PPI_SQL_FILE) \
		str str float int


clean:
	-rm *.tsv
