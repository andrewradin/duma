default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=cayman

include ../version_tools/setup.mk
include versions.mk

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

ZIP_FILE=CaymanStructureDefinitions.zip
RAW_FILE=CaymanStructureDefinitions.sdf
INPUT=$(WS_DNLD)/CaymanStructureDefinitions_$(CAYMAN_VER).sdf.gz
ARCHIVED_INPUTS=$(INPUT)
DNLD_URL=https://www.caymanchem.com/static/structure-definitions

ATTR_FILE=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv

OUTPUTS=\
	$(ATTR_FILE) \
	# end of list

TMP_IN=in.tmp
TMP_OUT=out.tmp

show_downloads:
	@echo $(INPUT)

input: $(INPUT)
build: $(OUTPUTS)

# The Cayman website now requires an account to download.
# Aaron has an account and can either download or share ass needed.
# With that in place now we download the file to a local machine and
# copy it over to the ETL machine. Cayman updates
# that data more-or-less continuously, so we can't know the
# update date in advance. Instead, the download make target below
# grabs the latest and labels the download file with the date. This
# date can then be used to set CAYMAN_VER in the versions.py file.
# If the file isn't present, we assume we're re-running a prior
# version and attempt to retrieve the file from S3.
show_latest_version:
	@echo
	@echo '======================================'
	@echo - get credentials from Aaron, and download:
	@echo '  $(ZIP_FILE) from https://www.caymanchem.com/structuredefinitions'
	@echo - place the file in this directory, and run:
	@echo make convert_download
	@echo 'This will report out the update for CAYMAN_VER in versions.py'
	@echo
	@echo 'After making that update, then:'
	@echo make input
	@echo make build
	@echo - manually verify changes are worth publishing
	@echo '  (if not, stop and revert versions.py changes)'
	@echo - run:
	@echo make publish_s3

old_download:
	wget $(DNLD_URL)/$(ZIP_FILE)
	$(MAKE) convert_download

convert_download:
	unzip $(ZIP_FILE) $(RAW_FILE)
	gzip $(RAW_FILE)
	CAYMAN_VER=$(shell ./cayman_date_extractor.sh $(ZIP_FILE) $(RAW_FILE))
	mv $(RAW_FILE).gz $(WS_DNLD)/CaymanStructureDefinitions_$(CAYMAN_VER).sdf.gz
	@echo "UPDATE: versions.py entry CAYMAN_VER=$(CAYMAN_VER)"
	rm $(ZIP_FILE)

$(INPUT):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

$(ATTR_FILE): $(INPUT)
	./parse_cayman.py -o $(TMP_OUT) $(INPUT)
	mv $(TMP_OUT) $(TMP_IN)
	../matching/make_std_smiles.py -r worker-test -i $(TMP_IN) -o $(TMP_OUT)
	mv $(TMP_OUT) $(ATTR_FILE)

clean:
	-rm *.tsv

