default:
	@echo "USAGE: make build|publish_ws|publish_s3|clean"

WS_TOX=$(shell ../../web1/path_helper.py tox)

CVT_PGM= ./process_gottlieb.py

# this file was hand-created based on data in the paper
INPUT=gottliebSideEffectsWithUMLS.tsv

OUT_ADR_FILENAME=gottliebSideEffectsWithMedDRA.tsv
OUT_ADR=$(WS_TOX)$(OUT_ADR_FILENAME)

OUTPUTS=\
	$(OUT_ADR_FILENAME) \
	# end of list

build: $(OUTPUTS)

publish_ws: $(OUTPUTS)
	cp $(OUT_ADR_FILENAME) $(OUT_ADR)

publish_s3: publish_ws
	../matching/move_s3_files.py --put tox $(OUT_ADR_FILENAME)

$(OUTPUTS):
	python $(CVT_PGM) $(INPUT) $(OUT_ADR_FILENAME)

clean:
	-rm $(OUTPUTS)

