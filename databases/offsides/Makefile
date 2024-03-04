default:
	@echo "USAGE: make input|build|publish_ws|publish_s3|clean"

WS_MAIN=$(shell ../../web1/path_helper.py storage)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_TOX=$(shell ../../web1/path_helper.py tox)

COLLECTION=offsides
CVT_PGM= ./parse_offsides.py
AGR_PGM= ../meddra/aggregate_adr_up_meddra.py
S3MOVE=../matching/move_s3_files.py
MEDDRA_FILENAME=meddra.v19.tsv
MEDDRA_FILE=$(WS_MAIN)$(MEDDRA_FILENAME)

$(MEDDRA_FILE):
	$(S3MOVE) storage $(MEDDRA_FILENAME)

URL=http://helix-web.stanford.edu/download/tatonetti/
DNLD_FILENAME=3003377s-offsides
ZIPFILE=$(DNLD_FILENAME).zip
INPUT=$(WS_DNLD)$(DNLD_FILENAME).tsv

$(INPUT):
	wget "$(URL)$(ZIPFILE)"
	unzip $(ZIPFILE)
	rm $(ZIPFILE)
	mv $(DNLD_FILENAME).tsv $(INPUT)

show_downloads:
	@echo $(INPUT)

input: $(INPUT)

OUT_ADR_FILENAME=adr.$(COLLECTION).odds_ratio_aggregated.tsv
OUT_ADR=$(WS_TOX)$(OUT_ADR_FILENAME)

OUTPUTS=\
	$(OUT_ADR_FILENAME) \
	# end of list

build: $(OUTPUTS)

publish_ws: $(OUTPUTS)
	cp $(OUT_ADR_FILENAME) $(OUT_ADR)

publish_s3: publish_ws
	../matching/move_s3_files.py --put tox $(OUT_ADR_FILENAME)

$(OUTPUTS): $(INPUT) $(MEDDRA_FILE)
	python $(CVT_PGM) --meddra-map ../umls/umls_to_meddra.tsv $(INPUT)
	python $(AGR_PGM) $(MEDDRA_FILE) adr.$(COLLECTION).odds_ratio.tsv

clean:
	-rm *.tsv

