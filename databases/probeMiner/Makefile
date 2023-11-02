default:
    @echo "USAGE: make input|build|publish|clean"

COLLECTION=probeMiner

S3_STORAGE=s3://duma-datasets/
ODIR=$(shell ../../web1/path_helper.py storage)
DDIR=$(shell ../../web1/path_helper.py downloads)

FN=probeminer_datadump
URL_PRE=https://probeminer.icr.ac.uk/$(FN)
ZIP=.zip
DNLD_URL=$(URL_PRE)$(ZIP)
ZIPPED_FN=$(FN)$(ZIP)
TMP_IN=$(FN).txt

OSUF=.tsv.gz
O1_PREF=$(COLLECTION)_create$(OSUF)
O2_PREF=$(COLLECTION)_data$(OSUF)

TMP_CREATE=tmp_create.tsv
TMP_DATA=tmp_data.tsv
TMP_DIR=$(ODIR)$(FN)
PRS_PRGM=./parse_probeMiner.py

OFILE_CREATE=probeMiner_create.tsv.gz
OFILE_DATA=probeMiner_data.tsv.gz
ODIR_CREATE=$(ODIR)$(OFILE_CREATE)
ODIR_DATA=$(ODIR)$(OFILE_DATA)

OUTPUTS=\
	$(OFILE_CREATE) \
	$(OFILE_DATA) \


PUBS=\
    $(ODIR_CREATE) \
	$(ODIR_DATA) \


input: $(TMP_IN)
build: $(OUTPUTS)
publish: $(PUBS)

$(TMP_IN):
	wget --no-check-certificate -O $(ZIPPED_FN) $(DNLD_URL)
	unzip $(ZIPPED_FN) -d $(TMP_DIR) 
	mv $(TMP_DIR)/$(FN)*.txt $(TMP_IN)

$(OUTPUTS): $(TMP_IN)
	$(PRS_PRGM) $(TMP_IN) -create_out $(TMP_CREATE) -data_out $(TMP_DATA)
	gzip -c $(TMP_CREATE) > $(OFILE_CREATE)
	gzip -c $(TMP_DATA) > $(OFILE_DATA)

$(PUBS): $(OUTPUTS)
	cp $(OFILE_CREATE) $(ODIR_CREATE)
	cp $(OFILE_DATA) $(ODIR_DATA)

publish_s3: $(PUBS)
	s3cmd put $(ODIR)$(OFILE_CREATE) $(S3_STORAGE)$(OFILE_CREATE)
	s3cmd put $(ODIR)$(OFILE_DATA) $(S3_STORAGE)$(OFILE_DATA)

clean:
	rm -rf *.tsv *.tsv *.log *gz tmp $(TMP_DIR)

