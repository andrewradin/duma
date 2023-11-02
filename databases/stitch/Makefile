default:
	@echo "USAGE: make input|build|clean"

WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_STRG=$(shell ../../web1/path_helper.py storage)

COLLECTION=stitch
CVT_PGM= ./process_stitch.sh

DNLD_URL=http://stitch.embl.de/download/
VERSION=v5.0

DPI_FILENAME=9606.protein_chemical.links.$(VERSION).tsv.gz
DPI_INPUT=$(WS_DNLD)/$(DPI_FILENAME)
$(DPI_INPUT):
	wget $(DNLD_URL)protein_chemical.links.$(VERSION)/$(DPI_FILENAME)
	mv $(DPI_FILENAME) $(DPI_INPUT)

INCHI_FILENAME=chemicals.inchikeys.$(VERSION).tsv.gz
INCHI_INPUT=$(WS_DNLD)/$(INCHI_FILENAME)

$(INCHI_INPUT):
	wget $(DNLD_URL)$(INCHI_FILENAME)
	mv $(INCHI_FILENAME) $(INCHI_INPUT)

CHEM_FILENAME=chemicals.$(VERSION).tsv.gz
CHEM_INPUT=$(WS_DNLD)/$(CHEM_FILENAME)

$(CHEM_INPUT):
	wget $(DNLD_URL)$(CHEM_FILENAME)
	mv $(CHEM_FILENAME) $(CHEM_INPUT)

MATCH_FILENAME=chemical.aliases.$(VERSION).tsv.gz
MATCH_INPUT=$(WS_DNLD)/$(MATCH_FILENAME)

$(MATCH_INPUT):
	wget $(DNLD_URL)$(MATCH_FILENAME)
	mv $(MATCH_FILENAME) $(MATCH_INPUT)

CONVERTER=$(WS_STRG)HUMAN_9606_Uniprot_data.tsv

DOWNLOAD_FILES=$(INCHI_INPUT) $(DPI_INPUT) $(CHEM_INPUT) $(MATCH_INPUT)
input: $(DOWNLOAD_FILES)

show_downloads:
	@echo $(DOWNLOAD_FILES)

OUT_DS=create.$(COLLECTION).full.tsv
OUT_DPI=dpi.$(COLLECTION).default.tsv
OUT_FLAT_DPI=dpi.$(COLLECTION)_flat.default.tsv

OUTPUTS=\
	$(OUT_DPI) \
	$(OUT_DS) \
# end of list

build: $(OUTPUTS)

$(OUTPUTS): $(DOWNLOAD_FILES) $(CONVERTER)
	python parse_stitch.py $(DPI_INPUT) $(CHEM_INPUT) $(INCHI_INPUT) $(MATCH_INPUT) $(CONVERTER)
	mv stitch.attr.tsv $(OUT_DS)
	mv stitch.dpi.tsv $(OUT_DPI)
	rm conv.tmp dpi.tmp

clean:
	-rm *.tsv
	-rm *.tmp

