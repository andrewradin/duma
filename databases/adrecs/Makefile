default:
	@echo "USAGE: make input|build|publish_ws|publish_s3|clean"

WS_MAIN=$(shell ../../web1/path_helper.py storage)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_TOX=$(shell ../../web1/path_helper.py tox)

COLLECTION=adrecs
CVT_PGM= ./parse_adrecs.py
AGR_PGM= ../meddra/aggregate_adr_up_meddra.py 
S3MOVE=../matching/move_s3_files.py
MEDDRA_FILENAME=meddra.v19.tsv
MEDDRA_FILE=$(WS_MAIN)$(MEDDRA_FILENAME)

$(MEDDRA_FILE):
	$(S3MOVE) storage $(MEDDRA_FILENAME)

URL=http://bioinf.xmu.edu.cn/ADReCS/download/

ADR_FILENAME=ADReCS_ADR_info
ADR_ZIP=$(ADR_FILENAME).zip
ADR_XML=$(ADR_FILENAME).xml
ADR_URL=$(URL)$(ADR_ZIP)
ADR_INPUT=$(WS_DNLD)/$(ADR_XML)

$(ADR_INPUT):
	wget "$(ADR_URL)"
	unzip $(ADR_ZIP)
	rm $(ADR_ZIP)
	mv $(ADR_XML) $(ADR_INPUT)

DRUG_FILENAME=ADReCS_Drug_info
DRUG_ZIP=$(DRUG_FILENAME).zip
DRUG_XML=$(DRUG_FILENAME).xml
DRUG_URL=$(URL)$(DRUG_ZIP)
DRUG_INPUT=$(WS_DNLD)/$(DRUG_XML)

$(DRUG_INPUT):
	wget "$(DRUG_URL)"
	unzip $(DRUG_ZIP)
	rm $(DRUG_ZIP)
	mv $(DRUG_XML) $(DRUG_INPUT)

INPUT=$(ADR_INPUT) $(DRUG_INPUT)

show_downloads:
	@echo $(INPUT)

input: $(INPUT)

OUT_DS=create.$(COLLECTION).full.tsv
OUT_ADR_FILENAME=adr.$(COLLECTION).portion_aggregated.tsv
OUT_ADR=$(WS_TOX)$(OUT_ADR_FILENAME)
OUT_TOX_FILENAME=tox.$(COLLECTION).full.tsv
OUT_TOX=$(WS_TOX)$(OUT_TOX_FILENAME)

OUTPUTS=\
	$(OUT_DS) \
	$(OUT_ADR_FILENAME) \
	$(OUT_TOX_FILENAME) \
	# end of list

build: $(OUTPUTS)

publish_ws: $(OUTPUTS)
	cp $(OUT_ADR_FILENAME) $(OUT_ADR)
	cp $(OUT_TOX_FILENAME) $(OUT_TOX)

publish_s3: publish_ws
	../matching/move_s3_files.py --put tox $(OUT_ADR_FILENAME)

$(OUTPUTS): $(INPUT) $(MEDDRA_FILE)
	python $(CVT_PGM) --adrs $(ADR_INPUT) --drugs $(DRUG_INPUT)
	python $(AGR_PGM) $(MEDDRA_FILE) adr.$(COLLECTION).portion.tsv

clean:
	-rm *.tsv

