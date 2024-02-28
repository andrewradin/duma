default:
	@echo "USAGE: make input|build|clean"

PRCS_PRGM=reconstruct_drpias.py
WS_PPI=$(shell ../../web1/path_helper.py ppi)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
SOURCE=drpias
S3_PREFIX=s3://duma-datasets/drpias/

IFILE_NAME=interaction.txt.gz
S3_IFILE=$(S3_PREFIX)$(IFILE_NAME)
INPUT=$(WS_DNLD)/$(IFILE_NAME)

# XXX Using this file rather than drpias/uniprot_id.txt.gz
# XXX increases the number of output interactions from 290K
# XXX to 419K.  This is at least partially because the
# XXX file below has almost 3500 Entrez ids mapped to 
# XXX multiple proteins, where uniprot_id.txt has only 58
# XXX (most of them YEAST proteins)
UNIPROT_FILE_NAME=HUMAN_9606_Protein_Entrez.tsv
S3_UFILE=s3://duma-datasets/$(UNIPROT_FILE_NAME)
UNIPROT=$(shell ../../web1/path_helper.py storage)/$(UNIPROT_FILE_NAME)

PPI_FILE=ppi.$(SOURCE).default.tsv
PPI_SQL_FILE=ppi.$(SOURCE).default.sqlsv

OUTPUTS=\
    $(PPI_FILE) \
	$(PPI_SQL_FILE) \
    # end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUT) $(UNIPROT)
build: $(OUTPUTS)

$(INPUT):
	s3cmd get $(S3_IFILE) $(INPUT)

$(UNIPROT):
	s3cmd get $(S3_UFILE) $(UNIPROT)
    
$(PPI_FILE): $(INPUT) $(UNIPROT)
	python $(PRCS_PRGM) -i $(INPUT) -u $(UNIPROT) > temp.tsv
	../matching/standardize_pi.py temp.tsv > temp2
	mv temp2 $@
	rm temp.tsv

$(PPI_SQL_FILE): $(PPI_FILE)
	../../web1/scripts/tsv_convert.py \
		-i $(PPI_FILE) -o $(PPI_SQL_FILE) \
		str str float int


publish_ws: $(WS_PPI)/$(PPI_FILE) $(WS_PPI)/$(PPI_SQL_FILE)

$(WS_PPI)/$(PPI_FILE): $(PPI_FILE)
	cp $(PPI_FILE) $(WS_PPI)

$(WS_PPI)/$(PPI_SQL_FILE): $(PPI_SQL_FILE)
	cp $(PPI_SQL_FILE) $(WS_PPI)

publish_s3: $(WS_PPI)/$(PPI_FILE) $(WS_PPI)/$(PPI_SQL_FILE)
	../matching/move_s3_files.py --put --gzip ppi $(PPI_FILE)
	../matching/move_s3_files.py --put --gzip ppi $(PPI_SQL_FILE)

clean:
	-rm *.tsv
