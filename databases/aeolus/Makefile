default:
	@echo "USAGE: make download|input|build|clean"

DBNAME=AEOLUS
MYSQL_USER=root
CURDIR=$(shell pwd)
SQL_SCRPT=$(CURDIR)/build_aeolus.sql

URL=http://datadryad.org/bitstream/handle/10255/dryad.105332/
IFILE=aeolus_v1.zip
IZIP_FILE=$(IFILE)?sequence=1
IDIR=$(shell ../../web1/path_helper.py downloads)
s3DIR=s3://duma-datasets/
ZIP_INPUT=$(IDIR)$(IFILE)
INPUT_DIR=$(IDIR)$(DBNAME)/

I_SCD=standard_case_drug.tsv
I_SCI=standard_case_indication.tsv
I_SCO=standard_case_outcome.tsv
I_SDOCT=standard_drug_outcome_contingency_table.tsv
I_SCOC=standard_case_outcome_category.tsv
I_SDOC=standard_drug_outcome_count.tsv
I_SDOD=standard_drug_outcome_drilldown.tsv
I_SUAC=standard_unique_all_case.tsv
I_SDOS=standard_drug_outcome_statistics.tsv
I_C=concept.tsv
I_V=vocabulary.tsv

INPUTS=\
	$(INPUT_DIR)$(I_SCD) \
	$(INPUT_DIR)$(I_SCI) \
	$(INPUT_DIR)$(I_SCO) \
	$(INPUT_DIR)$(I_SDOCT) \
	$(INPUT_DIR)$(I_SCOC) \
	$(INPUT_DIR)$(I_SDOC) \
	$(INPUT_DIR)$(I_SDOD) \
	$(INPUT_DIR)$(I_SUAC) \
	$(INPUT_DIR)$(I_SDOS) \
	$(INPUT_DIR)$(I_C) \
	$(INPUT_DIR)$(I_V) \
	# end of list

DUMPFILE=$(DBNAME).sql.gz

download: $(ZIP_INPUT)
input: $(INPUTS)
build: $(DUMPFILE)

$(ZIP_INPUT):
	curl $(URL)$(IZIP_FILE) > tmp
	mv tmp $(ZIP_INPUT)

$(INPUTS):
	unzip -d $(INPUT_DIR) $(ZIP_INPUT)

$(DUMPFILE): $(INPUTS)
	cd $(INPUT_DIR) && \
	mysql -u $(MYSQL_USER) --local-infile $(DBNAME) < $(SQL_SCRPT)
	mysqldump -u $(MYSQL_USER) $(DBNAME) | gzip > $(DUMPFILE)

publish_s3: build
	s3cmd put $(DUMPFILE) $(s3DIR)$(DUMPFILE)

clean:
	-rm -r *.tsv
