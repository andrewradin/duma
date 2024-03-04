default:
	@echo "USAGE: make build|publish_ws|publish_s3|clean"

WS_MAIN=$(shell ../../web1/path_helper.py storage)
WS_TOX=$(shell ../../web1/path_helper.py tox)

COLLECTION=sider
CVT_PGM= ./process_sider.py
AGR_PGM= ../meddra/aggregate_adr_up_meddra.py
S3MOVE=../matching/move_s3_files.py
MEDDRA_FILENAME=meddra.v19.tsv
MEDDRA_FILE=$(WS_MAIN)$(MEDDRA_FILENAME)

$(MEDDRA_FILE):
	$(S3MOVE) storage $(MEDDRA_FILENAME)

OUT_ADR_DFLT_FILENAME=adr.$(COLLECTION).default_aggregated.tsv
OUT_ADR_DFLT=$(WS_TOX)$(OUT_ADR_DFLT_FILENAME)
OUT_ADR_ODDS_FILENAME=adr.$(COLLECTION).odds_ratio_aggregated.tsv
OUT_ADR_ODDS=$(WS_TOX)$(OUT_ADR_ODDS_FILENAME)
OUT_ADR_PORTION_FILENAME=adr.$(COLLECTION).portion_aggregated.tsv
OUT_ADR_PORTION=$(WS_TOX)$(OUT_ADR_PORTION_FILENAME)

OUTPUTS=\
	$(OUT_ADR_DFLT_FILENAME) \
	$(OUT_ADR_ODDS_FILENAME) \
	$(OUT_ADR_PORTION_FILENAME) \
	# end of list

build: $(OUTPUTS)

publish_ws: $(OUTPUTS)
	cp $(OUT_ADR_DFLT_FILENAME) $(OUT_ADR_DFLT)
	cp $(OUT_ADR_ODDS_FILENAME) $(OUT_ADR_ODDS)
	cp $(OUT_ADR_PORTION_FILENAME) $(OUT_ADR_PORTION)

publish_s3: publish_ws
	../matching/move_s3_files.py --put tox $(OUT_ADR_DFLT_FILENAME)
	../matching/move_s3_files.py --put tox $(OUT_ADR_ODDS_FILENAME)
	../matching/move_s3_files.py --put tox $(OUT_ADR_PORTION_FILENAME)

$(OUTPUTS): $(MEDDRA_FILE)
	python $(CVT_PGM) --verbose
	python $(AGR_PGM) $(MEDDRA_FILE) adr.$(COLLECTION).default.tsv
	python $(AGR_PGM) $(MEDDRA_FILE) adr.$(COLLECTION).odds_ratio.tsv
	python $(AGR_PGM) $(MEDDRA_FILE) adr.$(COLLECTION).portion.tsv

clean:
	-rm *.tsv

