default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=gwas_cat


PARSE_PRGM=./parse_gwasCat.py

DNLD=$(shell ../../web1/path_helper.py downloads)

DAT_URL=https://www.ebi.ac.uk/gwas/api/search/downloads/alternative

OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

IFILE=$(FILE_CLASS).$(GWASCAT_DNLD_VER).inputdata.tsv
DAT_INPUT=$(DNLD)$(IFILE)

OFILE=$(OFILE_STEM).data.tsv.gz
STD_OFILE=$(OFILE_STEM).studies.tsv
FAILED_SNPS_FILE=$(OFILE_STEM).failed_snps.tsv
FAILED_STUDIES_FILE=$(OFILE_STEM).failed_studies.tsv

ARCHIVED_INPUTS=$(DAT_INPUT)

INPUTS=$(ARCHIVED_INPUTS)

OUTPUTS=\
	$(OFILE) \
	$(STD_OFILE) \
	$(FAILED_SNPS_FILE) \
	# end of list

show_downloads:
	@echo $(INPUTS)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

input: $(INPUTS)
build: $(OUTPUTS)

download_input:
	curl $(DAT_URL) > temp
	mv temp $(DAT_INPUT)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo GWASCAT_DNLD_VER=`date +%Y%m%d`
	@echo
	@echo -run:
	@echo make download_input
	@echo make build
	@echo publish_s3

$(OUTPUTS): $(INPUTS)
	$(PARSE_PRGM) $(DAT_INPUT) $(STD_OFILE) $(FAILED_SNPS_FILE) $(FAILED_STUDIES_FILE) > temp #also outputs filter.log
	awk '{print "|" $$0 "\t"}' $(FAILED_STUDIES_FILE) > fsf.tmp
	grep -F -v -f fsf.tmp temp > stripmaxpheno.tmp
	gzip -c stripmaxpheno.tmp > $(OFILE)
	rm temp *.tmp

clean:
	rm -rf *.tsv *.tsv.gz temp *log
