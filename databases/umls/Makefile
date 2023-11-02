default:
	@echo "USAGE: make input|build|publish|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=umls

DNLD=$(shell ../../web1/path_helper.py downloads)

ORIG_INPUT=$(DNLD)umls-$(UMLS_VER)-mrconso.zip
INFLATED_INPUT=$(DNLD)$(UMLS_VER).MRCONSO.RRF
MEDDRA_INPUT=$(DNLD)MRCONSO.meddra.$(UMLS_VER).RRF.gz
ICD_INPUT=$(DNLD)MRCONSO.icd9_icd10.$(UMLS_VER).RRF.gz

INPUTS=$(MEDDRA_INPUT) $(ICD_INPUT)
ARCHIVED_INPUTS=$(ORIG_INPUT)

MEDDRA_OFILE=$(FILE_CLASS).$(OUT_VER).to_meddra.tsv
ICD_OFILE=$(FILE_CLASS).$(OUT_VER).to_ICD9_or_ICD10.tsv

OUTPUTS=\
	$(MEDDRA_OFILE) \
	$(ICD_OFILE) \

show_downloads:
	@echo $(ORIG_INPUT)

input: $(INPUTS)

build: $(OUTPUTS)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

$(MEDDRA_INPUT) $(ICD_INPUT): $(ARCHIVED_INPUTS)
	unzip $(ORIG_INPUT)
	mv MRCONSO.RRF $(INFLATED_INPUT)
	grep -w MDR $(INFLATED_INPUT) | gzip > tmp
	mv tmp $(MEDDRA_INPUT)
	grep -w -E 'ICD10CM|ICD10|MTHICD10|ICD9CM|ICD9|MTHICD9' $(INFLATED_INPUT) | gzip > tmp
	mv tmp $(ICD_INPUT)


$(MEDDRA_OFILE): $(MEDDRA_INPUT)
	zcat $(MEDDRA_INPUT) | cut -f 1,11 -d'|' | tr "|" "\t" | uniq > temp
	mv temp $(MEDDRA_OFILE)

$(ICD_OFILE): $(ICD_INPUT)
# that is: UMLS CID, ICD version, ICD code, name of the ICD code
	zcat $(ICD_INPUT) | cut -f 1,12,14,15 -d'|' | tr "|" "\t" | uniq > temp
	mv temp $(ICD_OFILE)


show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Aaron has a Google sign-in associated with his twoXAR email to download the needed input files.'
	@echo 'Once you have that do the following:'
	@echo 'Go to https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html'
	@echo 'Download the latest MRCONSO.RRF,'
	@echo ' verifying that it is newer than what is listed in version.py'
	@echo 'After the zip file is downloaded locally, move it to ws/downloads on this machine.'
	@echo 'Then update UMLS_VER in versions.py with the appropriate new value (e.g. 2020AB)'
	@echo
	@echo 'Then:'
	@echo make input
	@echo make build
	@echo make publish_s3
	@echo

clean:
	rm -rf *.tsv temp
