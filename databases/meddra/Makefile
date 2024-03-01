default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=meddra

PARSE_PRGM=parse_medra.py

I_NAME=MedAscii_$(MEDDRA_VER)
IFILE=$(I_NAME).tgz
DNLD=$(shell ../../web1/path_helper.py downloads)

DNLD_DIR=$(DNLD)$(I_NAME)

ARCHIVE=$(DNLD)$(IFILE)
ARCHIVED_INPUTS=$(ARCHIVE)

LLT=$(DNLD_DIR)/llt_$(MEDDRA_VER).asc
MDHIER=$(DNLD_DIR)/mdhier_$(MEDDRA_VER).asc
INPUTS=$(LLT) $(MDHIER)

OFILE_NAME=$(FILE_CLASS).$(OUT_VER).tsv

OUTPUTS=\
	$(OFILE_NAME) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

process_download: $(DNLD_DIR)
	mv $(DNLD)$(I_NAME)/llt.asc $(LLT)
	mv $(DNLD)$(I_NAME)/mdhier.asc $(MDHIER)
	tar -czvf $(ARCHIVE) $(DNLD)$(I_NAME)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

$(INPUTS): $(ARCHIVE)
	tar -zxvmf $(ARCHIVE)

$(OUTPUTS): $(INPUT)
	python $(PARSE_PRGM) $(MDHIER) $(LLT) temp
	mv temp $(OFILE_NAME)

show_latest_version:
	@echo
	@echo "The raw data for MedDRA $(IFILE) is downloaded from"
	@echo " https://www.meddra.org/software-packages"
	@echo " using sign in information Aaron receives each release"
	@echo " (via our paid subscription)."
	@echo "Then to open the zipped file a password is retreived"
	@echo " from https://apps.meddra.org/selfservice/"
	@echo "This has to go through Aaron's email currently."
	@echo "Next the MedAscii file in the unzipped folder,"
	@echo " should be renamed to MedAscii_[MEDDRA_VERSION]"
	@echo "After that, the folder should be placed in 2xar/ws/downloads."
	@echo "Upon doing that, note the latest meddra version"
	@echo " and update versions.py accordingly."
	@echo
	@echo "Afer that:"
	@echo "make process_download"
	@echo "make input"
	@echo "make build"
	@echo "make publish_s3"

clean:
	-rm -rf *.tsv MedAscii*

