default:
	@echo "USAGE: make input|build|clean"

DB_NAME=humantfs
PARSE_PRGM=./genelist_to_drugcollection.py

V=1.01
URL=http://humantfs.ccbr.utoronto.ca/download/v_$(V)/
DFILE=TFs_Ensembl_v_$(V).txt
IFILE=$(DB_NAME)_$(V).tsv
UNI_FILE=HUMAN_9606_Protein_Ensembl.tsv

ODIR=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

DPI_FILE=dpi.$(DB_NAME).default.tsv
CREATE_FILE=create.$(DB_NAME).full.tsv

INPUT=$(DNLD)$(IFILE)
UNI_CONVERT=$(ODIR)$(UNI_FILE)

INPUTS=\
	$(INPUT) \
	$(UNI_CONVERT) \
	# end of list

OUTPUTS=\
	$(DPI_FILE) \
	$(CREATE_FILE) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

$(INPUT):
	curl $(URL)$(DFILE) > temp
	mv temp $(INPUT)

$(UNI_CONVERT):
	s3cmd get s3://duma-datasets/$(UNI_FILE) $(UNI_CONVERT)

$(OUTPUTS): $(INPUTS)
	$(PARSE_PRGM) $(INPUT) $(UNI_CONVERT) temp temp2 $(DB_NAME)
	mv temp $(DPI_FILE)
	mv temp2 $(CREATE_FILE)

clean:
	rm -rf *.tsv *.tsv.gz temp
