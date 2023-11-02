default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=duma

include ../version_tools/setup.mk
include versions.mk

COLLECTION=duma

WS_DNLD=$(shell ../../web1/path_helper.py downloads)


S3_DIR=s3://duma-datasets
DS_FILE_PRESTD=duma_prestd.tsv.tmp
DS_FILE=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
RAW_DPI_FILE=raw.dpi.duma.default.tsv
DPI_FILE=$(FILE_CLASS).default.$(OUT_VER).evidence.tsv
PROTMAP_VER=HUMAN_9606.$(UNIPROT_VER)

OUTPUTS=\
	$(DS_FILE) \
	$(DPI_FILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(DS_FILE_PRESTD) $(RAW_DPI_FILE)

build: $(OUTPUTS)

$(DS_FILE_PRESTD):
	aws s3 cp $(S3_DIR)/raw.create.duma.full.tsv $(DS_FILE_PRESTD)


$(DS_FILE): $(DS_FILE_PRESTD)
	../matching/make_std_smiles.py -i $(DS_FILE_PRESTD) -o $(DS_FILE)

$(RAW_DPI_FILE):
	aws s3 cp $(S3_DIR)/$(RAW_DPI_FILE) $(RAW_DPI_FILE)

$(DPI_FILE): $(RAW_DPI_FILE)
	../matching/map_uniprot_names.py $(RAW_DPI_FILE) $(DPI_FILE) $(PROTMAP_VER)

show_latest_version:
	@echo
	@echo '======================================'
	@echo
	@echo 'Go to the Drug Edits page on the platform.'
	@echo  "Using the latest version available (End version) compare to the DPI version right before last quarters' update."
	@echo  " This can be determined by looking at versions.py in matching and taking the n-1 version for the last description mentioning ETL update/refresh"
	@echo  'Address any issues and verify the latest data has been accepted and published.'
	@echo
	@echo 'Then run:'
	@echo 'make clean'
	@echo
	@echo 'Before the normal:'
	@echo 'make input'
	@echo 'make build'
	@echo 'make publish_s3'
	@echo

clean:
	-rm *.tsv *.tmp

