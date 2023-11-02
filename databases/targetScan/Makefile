default:
	@echo "USAGE: make input|build|clean"


FILE_CLASS=targetscan

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)

PRGM=./parse_targetscan.py

URL=http://www.targetscan.org/$(TARGETSCAN_VER)/$(TARGETSCAN_VER)_data_download/
URL_FILE=Predicted_Targets_Context_Scores.default_predictions.txt
ZIPD_FILE=$(URL_FILE).zip
IFILE=targetscan.$(TARGETSCAN_VER).context_scores.txt
DNLD_ZIPD_FILE=$(IFILE).zip

OFILE=targetscan.$(OUT_VER).context_scores_human9606_ensembl.tsv

IPATH=$(DNLD)$(IFILE)
ARCHIVED_INPUTS=$(IPATH)

INPUT=$(ARCHIVED_INPUTS)

INPUTS=\
	$(INPUT) \
	# end of list

OUTPUTS=\
	$(OFILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo 'Go to http://www.targetscan.org/ '
	@echo 'At the top of the page will be "Release X.X: <Date>"'
	@echo 'Verify if in versions.py TARGETSCAN_VER differs from:'
	@echo '"vert_XX", note the dropping of the "." from above.'
	@echo '  i.e. vert_72 for version 7.2'
	@echo
	@echo 'If it varies, create a new version and update:'
	@echo make download_input
	@echo make build
	@echo make publish_s3
	@echo
	@echo 'If not, update the last_checked variable in versions.py'

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

# This allows a manual download from the website to the current versioned
# filename. Only use if you're creating a new input version.
download_input:
	curl $(URL)$(ZIPD_FILE) > $(DNLD)$(DNLD_ZIPD_FILE)
	unzip $(DNLD)$(DNLD_ZIPD_FILE)
	mv $(URL_FILE) $(IPATH)

$(OUTPUTS): $(IPATH)
	$(PRGM) $(IPATH) > temp
	mv temp $(OFILE)

clean:
	rm -rf *.tsv *.txt
