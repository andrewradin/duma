default:
	@echo "USAGE: make input|build|clean"

DB_NAME=brenda
PARSE_PRGM=./parse_brenda.py
S3_BUCKET=s3://duma-datasets/

IFILE_PREF=brenda_download
IFILE=$(IFILE_PREF).txt
IZIP=$(IFILE_PREF).zip

ODIR=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

OUT_FILE_UNCOMPD=$(DB_NAME).human.tsv
OUT_FILE=$(OUT_FILE_UNCOMPD).gz

INPUT_ZIP=$(DNLD)$(IZIP)
INPUT_FILE=$(DNLD)$(IFILE)

INPUTS=\
	$(INPUT_FILE) \
	# end of list

OUTPUTS=\
	$(OUT_FILE) \
	# end of list

PUBLISHED=$(ODIR)$(OUT_FILE)

publish: $(PUBLISHED)
input: $(INPUTS)
build: $(OUTPUTS)

$(INPUT_ZIP):
	s3cmd get $(S3_BUCKET)$(IZIP) temp
	mv temp $(INPUT_ZIP)

$(INPUT_FILE): $(INPUT_ZIP)
	unzip $(INPUT_ZIP) -d $(DNLD)
	touch $(INPUT_FILE)

$(OUT_FILE_UNCOMPD): $(INPUTS)
	$(PARSE_PRGM) $(INPUT_FILE) temp
	mv temp $(OUT_FILE_UNCOMPD)

$(OUT_FILE): $(OUT_FILE_UNCOMPD)
	gzip -c $(OUT_FILE_UNCOMPD) > temp
	mv temp $(OUT_FILE) 

$(PUBLISHED): $(OUTPUTS)
	cp $(OUT_FILE) $(PUBLISHED)

publish_s3: $(PUBLISHED)
	s3cmd put $(PUBLISHED) $(S3_BUCKET)$(OUT_FILE)

clean:
	rm -rf *.tsv *.txt *.zip temp $(INPUT_ZIP) $(INPUT_FILE)
