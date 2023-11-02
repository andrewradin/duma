include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=orange_book

DNLD=$(shell ../../web1/path_helper.py downloads)

USE_CASE_URL=https://www.accessdata.fda.gov/scripts/cder/ob/results_patent.cfm
TODAY=$(shell date +%Y%m%d)
USE_CASE_DOWNLOAD_FILE=$(FILE_CLASS).$(TODAY).use_case.csv

USE_CASE_INPUT=$(DNLD)$(USE_CASE_DOWNLOAD_FILE)
OB_INPUT=$(DNLD)$(FILE_CLASS).$(TODAY).ob.zip

OUTPUTS=\
	$(FILE_CLASS).$(OUT_VER).products.txt \
	$(FILE_CLASS).$(OUT_VER).patent.txt \
	$(FILE_CLASS).$(OUT_VER).use_codes.csv \
	# end of list

show_latest_version:
	@echo
	@echo "Add a new versions.py entry containing:"
	@echo ORANGE_BOOK_VER=$(TODAY)
	@echo "Manually download the use case file by going to:"
	@echo $(USE_CASE_URL)
	@echo "and clicking the CSV button; then rename the downloaded file"
	@echo $(USE_CASE_DOWNLOAD_FILE)
	@echo "in the ws/downloads directory."
	@echo "Run make build to complete the download and extract files."
	@echo "If the files haven't changed substantially, you can back"
	@echo "out this version. Otherwise, make publish_s3 and check in."

DNLD_URL=https://www.fda.gov/media/76860/download

input: $(USE_CASE_INPUT) $(OB_INPUT)

$(USE_CASE_INPUT):
	@echo "PLEASE MANUALLY DOWNLOAD FILE"
	@echo "(run 'make show_latest_version' for details)"
	@false

$(OB_INPUT):
	curl -o $(OB_INPUT) $(DNLD_URL)

build: input
	cp $(USE_CASE_INPUT) $(FILE_CLASS).$(OUT_VER).use_codes.csv
	@for ROLE in products patent; do \
		unzip $(OB_INPUT) $$ROLE.txt; \
		touch $$ROLE.txt; \
		mv $$ROLE.txt $(FILE_CLASS).$(OUT_VER).$$ROLE.txt; \
	done

