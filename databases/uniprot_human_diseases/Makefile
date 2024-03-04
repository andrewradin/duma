default:
	@echo "USAGE: make download|build|publish|clean"

COLLECTION=uniprot

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

HD_PGM= ./process_uniprot_humdisease.py

HD_FILENAME=humdisease.txt
HD_INPUT=$(WS_DNLD)/$(HD_FILENAME)

$(HD_INPUT):
	wget "http://www.uniprot.org/docs/$(HD_FILENAME)"
	mv $(HD_FILENAME) $(HD_INPUT)

show_downloads:
	@echo $(HD_INPUT)

download: $(HD_INPUT)

OUT_HDP=../../web1/browse/static/disease_proteins.$(COLLECTION).txt

build: download
	python $(HD_PGM) --hd $(HD_INPUT) 

publish: build
	cp humdisease.out.tsv $(OUT_HDP)


clean:
	-rm *.tsv

