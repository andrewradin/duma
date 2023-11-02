default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=agr

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)

URL=https://fms.alliancegenome.org/download/
URL_FILE=DISEASE-ALLIANCE_COMBINED.tsv
IFILE=$(FILE_CLASS).$(AGR_VER).combined.tsv
IPATH=$(DNLD)$(IFILE)


DIS_FILE=$(FILE_CLASS).$(OUT_VER).human.tsv
MODEL_FILE=$(FILE_CLASS).$(OUT_VER).models.tsv

ARCHIVED_INPUTS=$(IPATH)

INPUT=$(ARCHIVED_INPUTS)

OUTPUTS=\
	$(DIS_FILE) \
	$(MODEL_FILE) \
	# end of list

show_downloads:
	@echo $(INPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain the latest AGR version.'
	@echo 'This can be retrieved from https://www.alliancegenome.org/'
	@echo
	@echo 'Also ensure the latest/appropriate version of'
	@echo 'Uniprot is listed'
	@echo
	@echo 'Then:'
	@echo make download_input
	@echo make build
	@echo make publish_s3


input: $(INPUT)
build: $(OUTPUTS)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

# This allows a manual download from the website to the current versioned
# filename. Only use if you're creating a new input version.
download_input:
	curl -f $(URL)$(URL_FILE) > $(IPATH)

$(OUTPUTS): $(IPATH)
	./parse_agr.py $(IPATH) $(UNIPROT_VER) temp1 temp2
	mv temp1 $(DIS_FILE)
	mv temp2 $(MODEL_FILE)

clean:
	rm -rf *.tsv *.txt *.log
