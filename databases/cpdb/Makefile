default:
	@echo "USAGE: make input|build|clean"

PRCS_PRGM=./parse_cpdb_ppi.py
WS_PPI=$(shell ../../web1/path_helper.py ppi)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_STRG=$(shell ../../web1/path_helper.py storage)
SOURCE=cpdb

URL='http://cpdb.molgen.mpg.de/download/'
RAW_PREFIX=ConsensusPathDB_human_PPI
RAW_ZIPPED=$(RAW_PREFIX).gz
RAW_FILE=$(RAW_PREFIX).tsv
INPUT=$(WS_DNLD)/$(RAW_FILE)

PPI_FILE=ppi.$(SOURCE).default.tsv
PPI_SQL_FILE=ppi.$(SOURCE).default.sqlsv
OUTPUTS=\
	$(PPI_FILE) \
	$(PPI_SQL_FILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUT)
build: $(OUTPUTS)

$(INPUT):
	wget $(URL)$(RAW_ZIPPED)
	zcat $(RAW_ZIPPED) > $(INPUT)
	rm $(RAW_ZIPPED)

# Now we finally process the data, and out put our end results
$(PPI_FILE): $(INPUT)
	$(PRCS_PRGM) -i $(INPUT) -m $(shell cut -f4 $(INPUT)\
	| sort -nr\
	| awk ' { if ($$1 != 'NA'){a[i++]=$$1}; } END { x=int((i+1)/2); if (x < (i+1)/2) print (a[x-1]+a[x])/2.0; else print a[x-1]; }')\
	 -u $(WS_STRG)HUMAN_9606_Uniprot_data.tsv > temp.tsv
	../matching/standardize_pi.py temp.tsv > temp2
	mv temp2 $@
	rm temp.tsv

$(PPI_SQL_FILE): $(PPI_FILE)
	../../web1/scripts/tsv_convert.py \
		-i $(PPI_FILE) -o $(PPI_SQL_FILE) \
		str str float int

publish_ws: $(WS_PPI)/$(PPI_FILE) $(WS_PPI/$(PPI_SQL_FILE)

$(WS_PPI)/$(PPI_FILE): $(PPI_FILE)
	cp $(PPI_FILE) $(WS_PPI)

$(WS_PPI)/$(PPI_SQL_FILE): $(PPI_SQL_FILE)
	cp $(PPI_SQL_FILE) $(WS_PPI)

publish_s3: $(WS_PPI)/$(PPI_FILE) $(WS_PPI)/$(PPI_SQL_FILE)
	../matching/move_s3_files.py --put --gzip ppi $(PPI_FILE)
	../matching/move_s3_files.py --put --gzip ppi $(PPI_SQL_FILE)

clean:
	-rm *.tsv
