default:
	@echo "USAGE: make input|build|publish_ws|publish_s3|clean"

WS_MAIN=$(shell ../../web1/path_helper.py storage)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)

COLLECTION=humanProtAtlas
CVT_PGM=./parse_hpa.py
S3MOVE=../matching/move_s3_files.py

URL=http://www.proteinatlas.org/download/
DFILE=proteinatlas.xml.gz

HPA_URL=$(URL)$(DFILE)
HPA_INPUT=$(WS_DNLD)/$(DFILE)

$(HPA_INPUT):
	wget $(HPA_URL)
	mv $(DFILE) $(HPA_INPUT)

INPUT=$(HPA_INPUT)

show_downloads:
	@echo $(INPUT)

input: $(INPUT)

OUT_FILENAME=$(COLLECTION).expression.tsv.gz
OUT=$(WS_MAIN)$(OUT_FILENAME)

OUTPUTS=\
	$(OUT_FILENAME) \
	# end of list

build: $(OUTPUTS)

publish_ws: $(OUTPUTS)
	cp $(OUT_FILENAME) $(OUT)

publish_s3: publish_ws
	../matching/move_s3_files.py --put storage $(OUT_FILENAME)

$(OUTPUTS): $(INPUT)
	$(CVT_PGM) $(HPA_INPUT) > temp
	gzip -c temp > $(OUT_FILENAME)

clean:
	-rm -rf *.tsv temp *.gz
