default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

# this relies on a hand-tuned patient registry file from the Myotonic Dystrophy Foundation


HERE=$(shell pwd)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)
WS_OUTPUT=$(shell ../../web1/path_helper.py storage)

SRC_NAME=MDF

DRUG_PREFIX=drug
INDI_PREFIX=indi
DEMO_PREFIX=demo

INDI_DEDUP=diseasesFrom$(SRC_NAME).dedup.txt
DEMO_DEDUP=demoFrom$(SRC_NAME).dedup.txt
DRUG_DEDUP=drugsFrom$(SRC_NAME).dedup.txt
PRSCD_FILES=\
	$(INDI_DEDUP) \
	$(DEMO_DEDUP) \
	$(DRUG_DEDUP) \

PRGM=./parse_mdf.py

MAT_PREF=clin_ev.FAERS+$(SRC_NAME)
MAT_FILES=\
	$(MAT_PREF).drug_mat.npz \
	$(MAT_PREF).drug_cols.txt \
	$(MAT_PREF).indi_mat.npz \
	$(MAT_PREF).indi_cols.txt \
	$(MAT_PREF).demo_mat.npz \
	# end of list
MATRIX_PRGM=./make_matrix_files.py

IFILE=select_MDF_patient_registry.tsv
MEDDRA_FILE=meddra.v19.tsv
MEDDRA=$(WS_OUTPUT)$(MEDDRA_FILE)
INPUTS=\
	$(IFILE) \
	$(MEDDRA) \
	# end of list
s3DIR=s3://duma-datasets/

input: $(INPUTS)

build: $(MAT_FILES)

$(MEDDRA):
	s3cmd get $(s3DIR)$(MEDDRA_FILE) $(MEDDRA)
$(IFILE):
	s3cmd get $(s3DIR)$(IFILE) $(IFILE)

$(PRSCD_FILES):$(INPUTS)
	$(PRGM) $(IFILE) $(MEDDRA) drug indi demo
	mv drug $(DRUG_DEDUP)
	mv indi $(INDI_DEDUP)
	mv demo $(DEMO_DEDUP)

$(MAT_FILES):$(PRSCD_FILES)
	$(MATRIX_PRGM) $(SRC_NAME)

publish: build
	for FILE in $(MAT_FILES); do \
		cp $$FILE $(WS_OUTPUT); \
	done

publish_s3: build
	for FILE in $(MAT_FILES); do \
		s3cmd put $$FILE $(s3DIR)$$FILE; \
	done

clean:
	rm -rf *.csv *.txt *.tmp *.tsv *npz
