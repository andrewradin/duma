default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=drugbank

include ../version_tools/setup.mk
include versions.mk

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

COLLECTION=drugbank
CVT_PGM= ./drugbank2tsv.py

# Drugbank full downloads are zipped XML files. The file in the zip archive
# is always named drugbank.xml regardless of version. The version can be
# determined by examining the xml header. Each download file has a
# unique integer id that is part of the download URL. We use that integer
# as the DRUGBANK_VER, and add it to the filename after we re-compress
# it (switching from zip to gzip). There's also an extraction date that's
# a little more informative than the integer id, but that requires downloading
# a json index file and using that to map between them.
#
# Run 'make show_latest_version' to get the DRUGBANK_VER number and date for
# the versions.py file.
RAW_ZIPPED=drugbank.xml.$(DRUGBANK_VER).zip
INPUT=$(RAW_ZIPPED)
LICENSE=<REMOVED>
DNLD_URL=https://portal.drugbank.com/downloads

show_latest_version:
	./show_latest_version.py $(LICENSE) $(DNLD_URL)

$(INPUT):
	curl -L -o $(RAW_ZIPPED).tmp -u $(LICENSE) $(DNLD_URL)/$(DRUGBANK_VER)
	mv $(RAW_ZIPPED).tmp $(RAW_ZIPPED)

show_downloads:
	@echo $(INPUT)

input: $(INPUT)

OUT_DS=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
OUT_DPI=$(FILE_CLASS).default.$(OUT_VER).evidence.tsv

OUTPUTS=\
	$(OUT_DS) \
	$(OUT_DPI) \
	# end of list

TMP_OUT=tmp.tsv
TMP_IN=tmp2.tsv

$(OUT_DS): $(INPUT)
	$(CVT_PGM) -i $(INPUT) drugset > $(TMP_OUT)
	mv $(TMP_OUT) $(TMP_IN)
	../matching/make_std_smiles.py -r worker-test -i $(TMP_IN) -o $(TMP_OUT)
	rm $(TMP_IN)
	mv $(TMP_OUT) $@

PROTMAP_VER=HUMAN_9606.$(UNIPROT_VER)
$(OUT_DPI): $(INPUT)
	$(CVT_PGM) -i $(INPUT) dpi > $(TMP_OUT)
	mv $(TMP_OUT) $(TMP_IN)
	../matching/map_uniprot_names.py $(TMP_IN) $(TMP_OUT) $(PROTMAP_VER)
	mv $(TMP_OUT) $(TMP_IN)
	../matching/standardize_pi.py --edits dpi_falsehoods.tsv \
			--pre-sort $(TMP_IN) > $(TMP_OUT)
	rm $(TMP_IN)
	mv $(TMP_OUT) $@

build: $(OUTPUTS)

clean:
	-rm *.tsv

