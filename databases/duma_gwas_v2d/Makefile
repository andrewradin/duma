default:
	@echo "USAGE: make vep|input|build|rescue|publish|publish_s3|clean|full_clean"

include ../version_tools/setup.mk
include versions.mk


FILE_CLASS=duma_gwas_v2d
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)
WS=$(shell ../../web1/path_helper.py storage)

##### VARIABLES ######
##### Programs used ######

PARSE_PRGM=./combine_gwas.py

##### General purpose directories ######
DNLD=$(shell ../../web1/path_helper.py downloads)

##### Individual DB inputs ######

FAIL_FILE_NAME=failed_snps.tsv
DATA_FILE_NAME=data.tsv.gz
STUDIES_FILE_NAME=studies.tsv

# DB inputs generally follow the same patterns, but not all have all data.
# - PHEWAS has no fails file, because all of the data is considered 'fails' due to 
#   none of the phewas SNPs having allele or MAF
# - UKBB has no fails file, because... ?

GRASP_DIR=$(WS)grasp/
GRASP_STEM=grasp.$(GRASP_VER)
GRASP_IFILE=$(GRASP_DIR)$(GRASP_STEM).$(DATA_FILE_NAME)
GRASP_FAIL_FILE=$(GRASP_DIR)$(GRASP_STEM).$(FAIL_FILE_NAME)
GRASP_STUDIES_FILE=$(GRASP_DIR)$(GRASP_STEM).$(STUDIES_FILE_NAME)

PHEWAS_DIR=$(WS)phewas/
PHEWAS_STEM=phewas.$(PHEWAS_VER)
PHEWAS_IFILE=$(PHEWAS_DIR)$(PHEWAS_STEM).$(DATA_FILE_NAME)
PHEWAS_STUDIES_FILE=$(PHEWAS_DIR)$(PHEWAS_STEM).$(STUDIES_FILE_NAME)

GWASCAT_DIR=$(WS)gwas_cat/
GWASCAT_STEM=gwas_cat.$(GWASCAT_VER)
GWASCAT_IFILE=$(GWASCAT_DIR)$(GWASCAT_STEM).$(DATA_FILE_NAME)
GWASCAT_FAIL_FILE=$(GWASCAT_DIR)$(GWASCAT_STEM).$(FAIL_FILE_NAME)
GWASCAT_STUDIES_FILE=$(GWASCAT_DIR)$(GWASCAT_STEM).$(STUDIES_FILE_NAME)

UKBB_DIR=$(WS)ukbb/
UKBB_STEM=ukbb.$(UKBB_VER)
UKBB_IFILE=$(UKBB_DIR)$(UKBB_STEM).data_filtered.tsv.gz
UKBB_STUDIES_FILE=$(UKBB_DIR)$(UKBB_STEM).$(STUDIES_FILE_NAME)

FINNGEN_DIR=$(WS)finngen/
FINNGEN_STEM=finngen.$(FINNGEN_VER)
FINNGEN_IFILE=$(FINNGEN_DIR)$(FINNGEN_STEM).data_filtered.tsv.gz
FINNGEN_STUDIES_FILE=$(FINNGEN_DIR)$(FINNGEN_STEM).$(STUDIES_FILE_NAME)

INPUTS=\
	$(GRASP_IFILE) \
	$(GRASP_FAIL_FILE) \
	$(GRASP_STUDIES_FILE) \
	$(GWASCAT_IFILE) \
	$(GWASCAT_FAIL_FILE) \
	$(GWASCAT_STUDIES_FILE) \
	$(PHEWAS_IFILE) \
	$(PHEWAS_STUDIES_FILE) \
	$(UKBB_IFILE) \
	$(UKBB_STUDIES_FILE) \
	$(FINNGEN_IFILE) \
	$(FINNGEN_STUDIES_FILE) \
	# end of list

##### OUTPUTS ######

# All studies from all input sources - some of these will be removed and/or have redundancy.
COMBINED_STUDIES=all_studies.$(OUT_VER).tsv

# Contains only the studies we have passing snps for.
FILTERED_STUDIES=$(OFILE_STEM).studies.tsv
# v2d* contain the same data, organized different.
# The archive.zip file is convenient for pulling out variants for a known list of studies.
# The data.tsv.gz file is convenient for iterating the full dataset.
COMBINED_V2D_ARCHIVE=$(OFILE_STEM).archive.zip
COMBINED_V2D=$(OFILE_STEM).data.tsv.gz

# none of the phewas SNPs have allele or MAF, so they are all technically failures
FAILED_FILES=\
	$(GRASP_FAIL_FILE) \
	$(GWASCAT_FAIL_FILE) \
	$(PHEWAS_IFILE) \
	# end of list

OUTPUTS=\
	$(FILTERED_STUDIES) \
	$(COMBINED_V2D) \
	$(COMBINED_V2D_ARCHIVE) \
	# end of list


##### phony's ######

input: $(INPUTS)
build: $(OUTPUTS)


##### RECIPES ######

##### inputs ######

show_latest_version:
	@echo
	@echo 'Ensure our latest versions for the following are updated in versions.py:'
	@echo '  grasp'
	@echo '  gwascat'
	@echo '  phewas'
	@echo '  ukbb'
	@echo '  finngen'
	@echo
	@echo 'Visit https://ftp.ncbi.nih.gov/snp/latest_release/VCF/'
	@echo '  to determine the latest version of the VCF'
	@echo '  and update VCF_VERSION in versions.py accordingly.'
	@echo '  e.g. GCF_000001405.38'
	@echo '  The . version on this could change - you can look up details about these here:'
	@echo '  https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.39/'
	@echo
	@echo
	@echo 'Finally:'
	@echo make input
	@echo make build
	@echo make publish_s3

# VCF file from the NCBI - ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF
# The . version on this could change - you can look up details about these here:
# https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.38/
# (If you look in the directory, there is also probably an old one corresponding to GRCh37)
VCF_ALL=$(VCF_VERSION).gz

$(VCF_ALL):
	wget ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF/$(VCF_ALL)

##### intermediates ######

# The rescue process takes all failed SNPs and sees if we can find better data for them in dbSNP.
# This can fix missing allele data, and sometimes maf/mac failures if we find a higher frequency value.
RESCUE_SNPS_VCF=rescue_snps.$(OUT_VER).vcf
RESCUED_DATA=rescued_data.$(OUT_VER).tsv

$(RESCUE_SNPS_VCF): $(FAILED_FILES) $(VCF_ALL)
	./do_vcf_subset.py -o $(RESCUE_SNPS_VCF) -v $(VCF_ALL) $(FAILED_FILES)

$(RESCUED_DATA): $(RESCUE_SNPS_VCF) $(FAILED_FILES) $(COMBINED_STUDIES)
	./rescue_snps.py --vcf $(RESCUE_SNPS_VCF) -o $(RESCUED_DATA) -f filter.log -s $(COMBINED_STUDIES) $(FAILED_FILES)

##### outputs ######

$(COMBINED_STUDIES): $(GRASP_STUDIES_FILE) $(PHEWAS_STUDIES_FILE) $(GWASCAT_STUDIES_FILE) $(UKBB_STUDIES_FILE) $(FINNGEN_STUDIES_FILE)
	./combine_studies.py -o $(COMBINED_STUDIES) $(GRASP_STUDIES_FILE) $(PHEWAS_STUDIES_FILE) $(GWASCAT_STUDIES_FILE) $(UKBB_STUDIES_FILE) $(FINNGEN_STUDIES_FILE)

$(COMBINED_V2D): $(INPUTS) $(RESCUED_DATA)
	time $(PARSE_PRGM) \
		$(GRASP_IFILE) \
		$(GWASCAT_IFILE) \
		$(UKBB_IFILE) \
		$(FINNGEN_IFILE) \
		$(RESCUED_DATA) \
		-o combined.tmp
	cat combined.tmp | gzip > $(COMBINED_V2D)
	rm combined.tmp

$(COMBINED_V2D_ARCHIVE): $(COMBINED_V2D)
	../duma_gwas/make_zip.sh $(COMBINED_V2D) $(COMBINED_V2D_ARCHIVE)

$(FILTERED_STUDIES): $(COMBINED_V2D_ARCHIVE) $(COMBINED_STUDIES)
	zipinfo -1 $(COMBINED_V2D_ARCHIVE) > studies_with_data.tsv
	# Platform code expects a header matching GRASP's format
	head -1 $(GRASP_STUDIES_FILE) > filtered_studies.tmp
	grep -Ff studies_with_data.tsv $(COMBINED_STUDIES) >> filtered_studies.tmp
	mv filtered_studies.tmp $(FILTERED_STUDIES)

##### clean ######

clean:
	rm -rf *.txt *.tsv *.tsv.gz temp tmp* *log rescue_vepd* *.tmp *.gz *.zip
