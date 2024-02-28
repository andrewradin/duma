default:
	@echo "USAGE: make vep|input|build|rescue|publish|publish_s3|clean|full_clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=duma_gwas
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

##### VARIABLES ######

##### Programs used ######

PARSE_PRGM=./combine_gwas.py
PREP_PRGM=./prepare_duma_for_vep.py
POST_VEP_PRGM=./update_gwas_with_vep.py
RESCUE_POST_VEP=./update_failed_snps_with_vep.py

##### General purpose directories ######
DNLD=$(shell ../../web1/path_helper.py downloads)
S3DIR=s3://duma-datasets/


##### VEP specific files and directories ######
# TODO when we update VEP next, we need to bake the vep version into the vep dir
# as it is right now, you can update the vep version and it won't try to install the latest code
# but just update the data dirs
VEP_GENOME_VERSION=GRCh$(VEP_UCSC_HG_VER)
VEP_GITHUB_BRANCH=postreleasefix/$(VEP_RELEASE_VERSION)
VEP_DIR=./ensembl-vep_$(VEP_RELEASE_VERSION)
VEP_CACHE_DIR_NAME=vep_cache/$(VEP_GENOME_VERSION)/
VEP_CACHE_DIR=$(VEP_DIR)/$(VEP_CACHE_DIR_NAME)
FASTA_NAME=homo_sapiens/$(VEP_RELEASE_VERSION)_$(VEP_GENOME_VERSION)/Homo_sapiens.$(VEP_GENOME_VERSION).dna.toplevel.fa
VEP_FASTA=$(VEP_CACHE_DIR)$(FASTA_NAME)
VEP_PRGM=$(VEP_DIR)/vep
VEP_INSTALLED=$(VEP_DIR)/Bio/EnsEMBL/Registry.pm
VEPS=\
	$(VEP_DIR) \
	$(VEP_INSTALLED) \
	$(VEP_FASTA) \
	# end of list


##### Other inputs ######

UNI_FILE_CLASS=uniprot
UNI_FILE=$(UNI_FILE_CLASS).HUMAN_9606.$(UNIPROT_VER).Uniprot_data.tsv
WS=$(shell ../../web1/path_helper.py storage)
UNI_CONVERT=$(WS)/$(UNI_FILE_CLASS)/$(UNI_FILE)

# VCF file from the NCBI - ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF
# The . version on this could change - you can look up details about these here:
# https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.38/
# (If you look in the directory, there is also probably an old one corresponding to GRCh37)
VCF_ALL=$(VCF_VERSION).gz
# This file is 10x smaller (and thus 10x faster to grep), and seems to have
# most of the RSIDs in it... consider using this one for testing.
# VCF_ALL=common_all_20180418.vcf.gz

##### Individual DB inputs ######

FAIL_FILE_NAME=failed_snps.tsv
DATA_FILE_NAME=data.tsv.gz
STUDIES_FILE_NAME=studies.tsv
# This really should be updated to deal with these different sources
# more programatically instead of the copy and paste approach I've taken to date

# We used to ensure these were made in this file, but now with them versioned I removed that code
# That means it's up to the user to ensure the correct versions exist

GRASP_DIR=$(WS)grasp/
INTER_GRASP_DIR=../grasp/
GRASP_STEM=grasp.$(GRASP_VER)
GRASP_IFILE=$(GRASP_DIR)$(GRASP_STEM).$(DATA_FILE_NAME)
GRASP_FAIL_FILE=$(INTER_GRASP_DIR)$(GRASP_STEM).$(FAIL_FILE_NAME)
GRASP_STUDIES_FILE=$(GRASP_DIR)$(GRASP_STEM).$(STUDIES_FILE_NAME)

PHEWAS_DIR=$(WS)phewas/
INTER_PHEWAS_DIR=../phewas/
PHEWAS_STEM=phewas.$(PHEWAS_VER)
PHEWAS_IFILE=$(PHEWAS_DIR)$(PHEWAS_STEM).$(DATA_FILE_NAME)
PHEWAS_FAIL_FILE=$(INTER_PHEWAS_DIR)$(PHEWAS_STEM).$(FAIL_FILE_NAME)
PHEWAS_STUDIES_FILE=$(PHEWAS_DIR)$(PHEWAS_STEM).$(STUDIES_FILE_NAME)

GWASCAT_DIR=$(WS)gwas_cat/
INTER_GWASCAT_DIR=../gwasCat/
GWASCAT_STEM=gwas_cat.$(GWASCAT_VER)
GWASCAT_IFILE=$(GWASCAT_DIR)$(GWASCAT_STEM).$(DATA_FILE_NAME)
GWASCAT_FAIL_FILE=$(INTER_GWASCAT_DIR)$(GWASCAT_STEM).$(FAIL_FILE_NAME)
GWASCAT_STUDIES_FILE=$(GWASCAT_DIR)$(GWASCAT_STEM).$(STUDIES_FILE_NAME)

UKBB_DIR=$(WS)ukbb/
UKBB_STEM=ukbb.$(UKBB_VER)
UKBB_IFILE=$(UKBB_DIR)$(UKBB_STEM).data_filtered.tsv.gz
UKBB_STUDIES_FILE=$(UKBB_DIR)$(UKBB_STEM).$(STUDIES_FILE_NAME)

##### Lists and combined intermediates ######

COMBINED_STUDIES=all_possible_studies.tsv
ALL_KEYS=all_possible_keys.tsv

INIT_FILES=\
	$(GRASP_IFILE) \
	$(GRASP_STUDIES_FILE) \
	# end of list

SECOND_IFILES=\
	$(GWASCAT_IFILE) \
	# end of list

SECOND_SFILES=\
	$(GWASCAT_STUDIES_FILE) \
	# end of list

SECOND_SFILE=tmp_studies.tsv
SECOND_IFILE=tmp_data.tsv.gz

INPUTS=\
	$(INIT_FILES) \
	$(SECOND_IFILE) \
	$(SECOND_SFILE) \
	$(UNI_CONVERT) \
	$(UKBB_IFILE) \
	$(UKBB_STUDIES_FILE) \
	# end of list

NO_RESCUE_OFILE=tmp_duma_gwas.tsv.gz
NO_RESCUE_STUDIES_FILE=tmp_duma_gwas_studies.tsv
INITIAL_OUTPUTS=\
	$(NO_RESCUE_OFILE) \
	$(NO_RESCUE_STUDIES_FILE) \
	# end of list


##### Rescue specifc files ######

COMBINE_FAIL=duma_$(FAIL_FILE_NAME).gz
# none of the phewas SNPs have allele or MAF, so they are all technically failures
FAILED_FILES=\
	$(GRASP_FAIL_FILE).gz \
	$(GWASCAT_FAIL_FILE).gz \
	$(COMBINE_FAIL) \
	$(PHEWAS_IFILE) \
	# end of list

ALL_FAILED=concatenated_$(FAIL_FILE_NAME).gz
ALL_FAILED_RSS=failed_rss.txt
TO_RESCUE=missing_failed_rss.txt
PREVIOUSLY_RESCUED=previous_rs_keyed_vep_results.for_$(OUT_VER).tsv.gz
RESCUED=rescued_SNPs.tsv.gz


##### Outputs ######

OFILE=$(OFILE_STEM).data.tsv.gz
OFILE_ZIP=$(OFILE_STEM).archivedata.zip
STUDIES_FILE=$(OFILE_STEM).studies.tsv
NEW_PREV_RESCD=$(OFILE_STEM).prev_rscd.tsv.gz

OUTPUTS=\
	$(OFILE) \
	$(OFILE_ZIP) \
	$(STUDIES_FILE) \
	$(NEW_PREV_RESCD) \
	# end of list


##### phony's ######

vep: $(VEPS)
input: $(INPUTS) $(VEPS)
rescue: $(RESCUED)
build: $(OUTPUTS)


##### RECIPES ######

##### inputs ######

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Visit https://github.com/Ensembl/ensembl-vep/releases'
	@echo '  to determine the latest version of VEP and update'
	@echo '  VEP_RELEASE_VERSION in versions.py'
	@echo 'NOTE: updating VEP will increase the run time significantly'
	@echo
	@echo 'Visit https://ftp.ncbi.nih.gov/snp/latest_release/VCF/'
	@echo '  to determine the latest version of the VCF that ends in .38'
	@echo '  and update VCF_VERSION in versions.py accordingly.'
	@echo '  e.g. GCF_000001405.38'
	@echo
	@echo 'Identify our latest version number for UCSC_HG (e.g. X) and update it for'
	@echo ' UCSC_HG_VER=X and VEP_UCSC_HG_VER=get_hg_v(X)'
	@echo
	@echo 'Then ensure our latest versions for the following are updated in versions.py:'
	@echo '  uniprot'
	@echo '  grasp'
	@echo '  gwascat'
	@echo '  phewas'
	@echo '  ukbb'
	@echo
	@echo 'Finally:'
	@echo make input
	@echo make build
	@echo make publish_s3
	@echo
	@echo "NOTE: It is suggested to run the make steps in a screen because some steps can take DAYS"
	@echo "NOTE2: The best way to speed this up is to bump up the ETL machine to include more cores"
	@echo "       If you do this, do it before make build which will take >4days w/4 cores"


$(VEP_DIR):
	git clone --single-branch --branch $(VEP_GITHUB_BRANCH) https://github.com/Ensembl/ensembl-vep.git
	mv ensembl-vep $(VEP_DIR)
	mkdir -p $(VEP_CACHE_DIR)

$(VEP_INSTALLED): $(VEP_DIR)
	cd $(VEP_DIR) && \
	perl INSTALL.pl --AUTO a --NO_HTSLIB --NO_UPDATE
	touch $(VEP_INSTALLED)

$(VEP_FASTA): $(VEP_INSTALLED)
	cd $(VEP_DIR) && \
	perl INSTALL.pl \
	--AUTO cf --SPECIES homo_sapiens \
	--ASSEMBLY $(VEP_GENOME_VERSION) \
	--CACHEDIR $(VEP_CACHE_DIR_NAME) \
	--NO_UPDATE
	-gunzip $(VEP_FASTA).gz
	touch $(VEP_FASTA)

$(VCF_ALL):
	wget ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF/$(VCF_ALL)

$(UNI_CONVERT):
	 $(S3_TOOL) $(UNI_BUCKET) $(UNI_FILE)

$(SECOND_IFILE):$(SECOND_IFILES)
	zcat $(SECOND_IFILES) | sort -k1,1 | gzip -c > tmp
	mv tmp $(SECOND_IFILE)

$(SECOND_SFILE): $(SECOND_SFILES)
	for f in $(SECOND_SFILES); do tail -n+2 $$f | sort -k1,1 >> tmp; done
	mv tmp $(SECOND_SFILE)

NPROC := $(shell nproc)

##### intermediates ######

$(INITIAL_OUTPUTS) $(FAILED_FILES): $(INPUTS) $(VEPS)
	time $(PARSE_PRGM) \
	$(INIT_FILES) \
	$(NO_RESCUE_STUDIES_FILE) \
	-s $(SECOND_SFILE) \
	$(UKBB_STUDIES_FILE) \
	-i $(SECOND_IFILE) \
	$(UKBB_IFILE) | gzip > raw_combined.gz
	time $(PREP_PRGM) raw_combined.gz temp.gz
	time zcat temp.gz | sort -V -k1,1 -k2,2 | gzip > for_vep.temp.gz
	rm temp.gz
	mv for_vep.temp.gz for_vep.gz
	time perl $(VEP_PRGM) \
	-i for_vep.gz \
	-o vepd.gz \
	--tab \
	--dir_cache $(VEP_CACHE_DIR) \
	--fasta $(VEP_FASTA) \
	--cache \
	--force_overwrite \
	--compress_output gzip \
	--fork $(NPROC) \
	--buffer_size 100000 \
	--max_af \
	--offline \
	--lookup_ref
	time $(POST_VEP_PRGM) $(UNI_CONVERT) vepd.gz raw_combined.gz temp.gz $(COMBINE_FAIL)
	-gzip $(GWASCAT_FAIL_FILE)
	-gzip $(GRASP_FAIL_FILE)
	# Even if we don't regzip, make sure the timestamps are updated
	touch $(GWASCAT_FAIL_FILE).gz
	touch $(GRASP_FAIL_FILE).gz
	touch $(PHEWAS_IFILE)
	mv temp.gz $(NO_RESCUE_OFILE)

$(COMBINED_STUDIES): $(INPUTS)
	cat $(GRASP_STUDIES_FILE) $(PHEWAS_STUDIES_FILE) $(GWASCAT_STUDIES_FILE) $(UKBB_STUDIES_FILE) | sort > temp
	mv temp $(COMBINED_STUDIES)

$(ALL_KEYS): $(COMBINED_STUDIES)
	cut -f 1 $(COMBINED_STUDIES) | sort -u > temp
	mv temp $(ALL_KEYS)


##### rescues ######

$(ALL_FAILED): $(ALL_KEYS) $(FAILED_FILES)
	zcat $(FAILED_FILES) | gzip > all_failed.tsv.gz
	zcat all_failed.tsv.gz | cut -f 1 | sort -u | gzip > all_failed_keys.tsv.gz
	zcat all_failed_keys.tsv.gz | comm -12 $(ALL_KEYS) - | gzip > failed_snp_keys_possible.tsv.gz
	bash -c "grep -F -f <(zcat failed_snp_keys_possible.tsv.gz) <(zcat all_failed.tsv.gz) | gzip -c > temp.gz"
	-rm all_failed.tsv.gz all_failed_keys.tsv.gz failed_snp_keys_possible.tsv.gz
	mv temp.gz $(ALL_FAILED)

$(ALL_FAILED_RSS): $(ALL_FAILED)
	zcat $(ALL_FAILED) | \
	cut -f 2 | \
	sort -u | \
	perl -lane 'print "rs".$$F[0];' > temp
	mv temp $(ALL_FAILED_RSS)

$(PREVIOUSLY_RESCUED):
	./get_prev_rescd_rss.py $(OUT_VER) $(PREVIOUSLY_RESCUED)

$(TO_RESCUE): $(ALL_FAILED_RSS) | $(PREVIOUSLY_RESCUED)
	zcat $(PREVIOUSLY_RESCUED) | perl -lane 'print if $$F[0] !~ "##";' | cut -f 1 | sort -u > tmp
	comm -13 tmp $(ALL_FAILED_RSS) > temp
	rm tmp
	mv temp $(TO_RESCUE)

$(NEW_PREV_RESCD): $(TO_RESCUE) $(VCF_ALL) | $(PREVIOUSLY_RESCUED)
	# Generate a VCF file from the rs ids.
	# This takes a huge VCF file from the NCBI and subsets it to our IDs
	# This allows us to run VEP in offline mode (rs ids are not supported)
	time ./rs_to_vcf.sh $(TO_RESCUE) $(VCF_ALL) tmp.$(TO_RESCUE).vcf
	time perl $(VEP_PRGM) \
	-i tmp.$(TO_RESCUE).vcf \
	--format vcf \
	-o rescue_vepd \
	--tab \
	--dir_cache $(VEP_CACHE_DIR) \
	--fasta $(VEP_FASTA) \
	--force_overwrite \
	--cache \
	--fork $(NPROC) \
	--buffer_size 100000 \
	--max_af \
	--offline
	mv rescue_vepd temp_rescd
# no header as it's already there
# the "-" is to ensure this works even if the file is empty,
# as will happen whenever we upgrade VEP, the VCF file, or UCSC HG (i.e. often)
	-zcat $(PREVIOUSLY_RESCUED) | grep -v '#' - >> temp_rescd
	gzip temp_rescd
	mv temp_rescd.gz $(NEW_PREV_RESCD)

$(RESCUED): $(NEW_PREV_RESCD)
	$(RESCUE_POST_VEP) $(UNI_CONVERT) $(NEW_PREV_RESCD) $(ALL_FAILED) $(COMBINED_STUDIES) temp
	gzip -c temp > $(RESCUED)
	rm -f temp


##### final outputs ######

$(OFILE): $(RESCUED) $(NO_RESCUE_OFILE)
	cat $(RESCUED) $(NO_RESCUE_OFILE) > temp
	mv temp $(OFILE)

$(OFILE_ZIP): $(OFILE)
	./make_zip.sh $(OFILE) $(OFILE_ZIP).tmp
	mv $(OFILE_ZIP).tmp $(OFILE_ZIP)

$(STUDIES_FILE): $(RESCUED)
	zcat $(RESCUED) | cut -f 1 | sort -u > rescued_keys.tsv
	cat $(NO_RESCUE_STUDIES_FILE) | cut -f 1 | sort -u > no_rescued_keys.tsv
	comm -23 rescued_keys.tsv no_rescued_keys.tsv > new_keys.tsv
	# This grep could fail if there are no new keys.
	-grep -w -F -f new_keys.tsv $(COMBINED_STUDIES) > temp
	cat $(NO_RESCUE_STUDIES_FILE) > studies_file.tmp
	cat temp >> studies_file.tmp
	zcat $(OFILE) | cut -f 1 | sort -u > snp_studies.tmp
	# Pull out the header line
	head -1 studies_file.tmp > studies_file_0snpRemoved.tmp
	grep -w -F -f snp_studies.tmp studies_file.tmp >> studies_file_0snpRemoved.tmp
	rm rescued_keys.tsv new_keys.tsv temp snp_studies.tmp studies_file.tmp
	mv studies_file_0snpRemoved.tmp $(STUDIES_FILE)


##### clean ######

clean:
	rm -rf *.txt *.tsv *.tsv.gz temp tmp* *log rescue_vepd* for_vep raw_combined.gz vepd* *.tmp *.gz ensembl-vep/

full_clean:
	make clean
	cd $(GRASP_DIR) && make clean
	cd $(GWASCAT_DIR) && make clean
	cd $(PHEWAS_DIR) && make clean
	cd $(UKBB_DIR) && make clean
