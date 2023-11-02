default:
	@echo "USAGE: make input|build|clean"

WS_DNLD=$(shell ../../web1/path_helper.py downloads)
ODIR=$(shell ../../web1/path_helper.py storage)

COLLECTION=broad

CVT_PGM=./parse_broad_compds.py

URL=https://s3.amazonaws.com/data.clue.io/repurposing/downloads/

DPI_FILE=repurposing_drugs_20170327.txt
DPI_URL=$(URL)$(DPI_FILE)
DPI_INPUT=$(WS_DNLD)$(DPI_FILE)

$(DPI_INPUT):
	wget "$(DPI_URL)"
	mv $(DPI_FILE) $(DPI_INPUT)

CMPD_FILE=repurposing_samples_20170327.txt
CMPD_URL=$(URL)$(CMPD_FILE)
CMPD_INPUT=$(WS_DNLD)$(CMPD_FILE)

$(CMPD_INPUT):
	wget "$(CMPD_URL)"
	mv $(CMPD_FILE) $(CMPD_INPUT)


UNI_FILE=HUMAN_9606_Uniprot_data.tsv
ENSEMBL_UNIPROT_CONVERTER=$(ODIR)/$(UNI_FILE)

$(ENSEMBL_UNIPROT_CONVERTER):
	s3cmd get s3://duma-datasets/$(UNI_FILE) $(ENSEMBL_UNIPROT_CONVERTER)

INPUT=$(CMPD_INPUT) $(DPI_INPUT) $(ENSEMBL_UNIPROT_CONVERTER)

show_downloads:
	@echo $(INPUT)

input: $(INPUT)

OUT_DS=create.$(COLLECTION).full.tsv
OUT_DPI=dpi.$(COLLECTION).default.tsv

OUTPUTS=\
	$(OUT_DS) \
	$(OUT_DPI) \
	# end of list

build: $(OUTPUTS)

$(OUTPUTS): $(INPUT)
	python $(CVT_PGM) $(CMPD_INPUT) $(DPI_INPUT) $(ENSEMBL_UNIPROT_CONVERTER)
	mv broad $(OUT_DS)
	mv dpi $(OUT_DPI)

clean:
	-rm *.tsv broad dpi
