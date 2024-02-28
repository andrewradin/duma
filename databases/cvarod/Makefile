default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

URL=https://www.canada.ca/content/dam/hc-sc/migration/hc-sc/dhp-mps/alt_formats/zip/medeff/databasdon/extract_extrait.zip
HERE=$(shell pwd)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_OUTPUT=$(shell ../../web1/path_helper.py storage)
ZIPFILE=$(WS_DNLD)extract_extrait.zip

DATA_DIR=data

# Unlike FAERS, which publishes incremental files, CVAROD updates a
# single cumulative zip file each month. So, just extract it locally
# rather than setting aside a ws subdirectory for archiving past files.

DRUG_PREFIX=drug
INDI_PREFIX=indi
DEMO_PREFIX=demo

INDI_ACCUM=diseasesFromCVAROD_unmapped.txt
INDI_MAPPED=diseasesFromCVAROD.txt
INDI_DEDUP=diseasesFromCVAROD.dedup.txt

DEMO_ACCUM=demoFromCVAROD.txt
DEMO_DEDUP=demoFromCVAROD.demo_dedup.txt

DRUG_ACCUM=drugsFromCVAROD_unmapped.txt
DRUG_MAPPED=drugsFromCVAROD.txt
DRUG_DEDUP=drugsFromCVAROD.dedup.txt

DATE_ACCUM=dateFromCVAROD.txt
DATE_DEDUP=dateFromCVAROD.date_dedup.txt

PARSE_DATA_PGRM=./parseCVARODdata.sh
DRUG_MAP_PRGM=../faers/map_drugs.py
INDI_MAP_PRGM=../faers/map_indis.py

DEDUP_PRGM=../faers/deduplicate.py
DEMO_DEDUP_PRGM=../faers/demo_deduplicate.py --cvarod
DATE_DEDUP_PRGM=../faers/date_deduplicate.py --cvarod

MAT_FILES=\
	clin_ev.CVAROD.drug_mat.npz \
	clin_ev.CVAROD.drug_cols.txt \
	clin_ev.CVAROD.indi_mat.npz \
	clin_ev.CVAROD.indi_cols.txt \
	clin_ev.CVAROD.demo_mat.npz \
	clin_ev.CVAROD.date_mat.npz \
	# end of list
MATRIX_PRGM=../faers/make_matrix_files.py

ACCUM_DATA=\
	$(DRUG_ACCUM) \
	$(INDI_ACCUM) \
	$(DEMO_ACCUM) \
	$(DATE_ACCUM) \
	# end of list

INPUTS=\
	$(DRUG_DEDUP) \
	$(INDI_DEDUP) \
	$(DEMO_DEDUP) \
	$(DATE_DEDUP) \
	# end of list
s3DIR=s3://duma-datasets/

.PHONY: input clean build

modified:
	# this sets the date of the file 'modified' to match the
	# modified date of the download file
	touch -d "`wget --spider -S $(URL) 2>&1 | sed -n 's/.*Modified: //p'`" modified
	$(MAKE) $(ZIPFILE)

input: $(ZIPFILE)

$(ZIPFILE): modified
	rm -rf data
	@echo ADR database is updated. Downloading.
	-rm -f $@
	wget --no-use-server-timestamps $(URL) -P $(WS_DNLD)
	unzip -o $@ -d data

build: $(MAT_FILES)

$(ACCUM_DATA): $(ZIPFILE)
	-rm -f $(INDI_ACCUM) $(DRUG_ACCUM) $(DEMO_ACCUM)
	$(PARSE_DATA_PGRM) \
		$(HERE)/$(INDI_ACCUM) \
		$(HERE)/$(DRUG_ACCUM) \
		$(HERE)/$(DEMO_ACCUM) \
		$(HERE)/$(DATE_ACCUM) \
		$(DATA_DIR)

$(DRUG_MAPPED): $(DRUG_ACCUM)
	$(DRUG_MAP_PRGM) $(DRUG_ACCUM) $(DRUG_MAPPED)

$(INDI_MAPPED): $(INDI_ACCUM)
	$(INDI_MAP_PRGM) $(INDI_ACCUM) $(INDI_MAPPED)

%.dedup.txt:%.txt
	$(DEDUP_PRGM) $*.txt

%.demo_dedup.txt:%.txt
	$(DEMO_DEDUP_PRGM) $*.txt

%.date_dedup.txt:%.txt
	$(DATE_DEDUP_PRGM) $*.txt

$(MAT_FILES):$(INPUTS)
	$(MATRIX_PRGM) CVAROD

publish: build
	for FILE in $(MAT_FILES); do \
		cp $$FILE $(WS_OUTPUT); \
	done

publish_s3: build
	for FILE in $(MAT_FILES); do \
		s3cmd put $$FILE $(s3DIR)$$FILE; \
	done

clean:
	rm -f *.txt
