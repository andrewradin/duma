default:
	@echo "USAGE: make input|build|clean"

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

COLLECTION=ttd
CVT_PGM= ./process_TTD_DPI_data.py

URL_PREFIX=https://db.idrblab.org/ttd/sites/default/files/ttd_database

SDF_FILENAME=P3-01-All.sdf
SDF_INPUT=$(WS_DNLD)/TTD_$(SDF_FILENAME)

$(SDF_INPUT):
	wget "$(URL_PREFIX)/$(SDF_FILENAME)"
	mv $(SDF_FILENAME) $(SDF_INPUT)

DPI_FILENAME=P1-01-TTD_download.txt
DPI_INPUT=$(WS_DNLD)/$(DPI_FILENAME)

$(DPI_INPUT):
	wget "$(URL_PREFIX)/$(DPI_FILENAME)"
	tail -n+13 $(DPI_FILENAME) > $(DPI_INPUT)
	rm $(DPI_FILENAME)

MATCH_FILENAME=P1-02-TTD_crossmatching.txt
MATCH_INPUT=$(WS_DNLD)/$(MATCH_FILENAME)

$(MATCH_INPUT):
	wget "$(URL_PREFIX)/$(MATCH_FILENAME)"
	tail -n+13 $(MATCH_FILENAME) > $(MATCH_INPUT)
	rm $(MATCH_FILENAME)

INPUT=$(SDF_INPUT) $(DPI_INPUT) $(MATCH_INPUT)

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
	$(CVT_PGM) --dpi $(DPI_INPUT) --attr --struct $(SDF_INPUT) \
				--filter $(MATCH_INPUT)
	mv dpi.out.tsv $(OUT_DPI)
	mv attr.out.tsv $(OUT_DS)

clean:
	-rm *.tsv

