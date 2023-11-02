default:
	@echo "USAGE: make vep|input|build|rescue|publish|publish_s3|clean|full_clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=duma_gwas_v2g
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

##### VARIABLES ######

##### Programs used ######

PREP_PRGM=./prepare_duma_for_vep.py
POST_VEP_PRGM=./vep_out_to_v2g.py

##### General purpose directories ######
DNLD=$(shell ../../web1/path_helper.py downloads)
S3DIR=s3://duma-datasets/


##### VEP specific files and directories ######
VEP_GENOME_VERSION=GRCh$(VEP_UCSC_HG_VER)
VEP_GITHUB_BRANCH=release/$(VEP_RELEASE_VERSION)
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

INPUT_SNPS=$(WS)duma_gwas_v2d/duma_gwas_v2d.v$(DUMA_GWAS_V2D_VER).data.tsv.gz

INPUTS=\
	$(INPUT_SNPS) \
	# end of list

##### Outputs ######

# Final outputs
OFILE=$(OFILE_STEM).data.tsv.gz
OFILE_ZIP=$(OFILE_STEM).archive.zip
OFILE_D2G_SUMMARY=$(OFILE_STEM).d2g_summary.sqlsv
OFILE_OTARG_ALLELES=$(OFILE_STEM).otarg_alleles.sqlsv


OUTPUTS=\
	$(OFILE) \
	$(OFILE_ZIP) \
	$(OFILE_D2G_SUMMARY) \
	$(OFILE_OTARG_ALLELES) \
	# end of list


##### phony's ######

vep: $(VEPS)
input: $(INPUTS) $(VEPS)
build: $(OUTPUTS)


##### RECIPES ######


show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Visit https://github.com/Ensembl/ensembl-vep/branches'
	@echo '  to determine the latest default version e.g. release/108 of VEP and update'
	@echo '  VEP_RELEASE_VERSION in versions.py'
	@echo 'NOTE: updating VEP will increase the run time significantly'
	@echo
	@echo 'Update OTARG_GEN_VERSION by finding the latest directory name here:'
	@echo ' http://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/'
	@echo
	@echo 'Identify our latest version number for UCSC_HG (e.g. X) and update it for'
	@echo ' UCSC_HG_VER=X and VEP_UCSC_HG_VER=get_hg_v(X)'
	@echo
	@echo 'Then ensure our latest versions for the following are updated in versions.py:'
	@echo '  uniprot'
	@echo
	@echo 'Finally:'
	@echo 'make input - though note this may not be necessary depending on which versions were updated'
	@echo make build
	@echo make publish_s3
	@echo
	@echo "NOTE: It is suggested to run the make steps in a screen because some steps can take a while"


##### OpenTargets pipeline ######

# Download data from opentargets
OTARG_V2G_INPUT=otarg_v2g_input.$(OTARG_GEN_VERSION)
$(OTARG_V2G_INPUT):
	rm -rf tmp.$(OTARG_V2G_INPUT)
	./otarg_download_v2g.sh $(OTARG_GEN_VERSION) tmp.$(OTARG_V2G_INPUT)
	mv tmp.$(OTARG_V2G_INPUT) $(OTARG_V2G_INPUT)

# Subset to find our snps of interest
OTARG_SNP_SUBSET=otarg_v2g_snp_subset.$(OTARG_GEN_VERSION)
$(OTARG_SNP_SUBSET): $(OTARG_V2G_INPUT) $(INPUT_SNPS)
	rm -rf tmp.$(OTARG_SNP_SUBSET)
	mkdir -p tmp.$(OTARG_SNP_SUBSET)
	./otarg_snp_search.py \
		--input-variants $(INPUT_SNPS) \
		--output tmp.$(OTARG_SNP_SUBSET)/snps \
		--fail-file otarg.missing_variants.tsv \
		$(OTARG_V2G_INPUT)/*.zst
	rm -rf $(OTARG_SNP_SUBSET)
	mv tmp.$(OTARG_SNP_SUBSET) $(OTARG_SNP_SUBSET)

# Aggregate the snp->gene scores
OTARG_SNP_OFILE=otarg_snps.tsv.gz
$(OTARG_SNP_OFILE) $(OFILE_OTARG_ALLELES): $(OTARG_SNP_SUBSET)
	./otarg_snp_parse.py \
		--output-v2g tmp.$(OTARG_SNP_OFILE) \
		--output-alleles $(OFILE_OTARG_ALLELES) \
		-u $(UNI_CONVERT) \
		$(OTARG_SNP_SUBSET)/*
	
	mv tmp.$(OTARG_SNP_OFILE) $(OTARG_SNP_OFILE)


##### vep setup ######

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

$(UNI_CONVERT):
	 $(S3_TOOL) $(UNI_BUCKET) $(UNI_FILE)

NPROC := $(shell nproc)

##### vep pipeline ######

# Raw vep output
VEP_OUTPUT=vepd.$(OUT_VER).gz
# vep output converted to our snp format.
VEP_SNP_OFILE=snps.$(OUT_VER).tsv.gz

# Runs VEP on all variant coordinates provided by v2d, to get associated genes.
# === Other interesting VEP options ===
# --lookup_ref will ignore our provided reference allele and look it u in the fasta file
#   (must also provide --fasta $(VEP_FASTA))
#   This will make a lot of the SNPs fail to process, due to mismatched ref/alt
# --no_check_alleles will ignore the allele provided and just look at position.  Maybe we should use?
# --max_af will lookup the allele frequency from several integrated DBs - we do this on the v2d side via dbSNP now
$(VEP_OUTPUT): $(INPUT_SNPS) $(VEPS)
	time $(PREP_PRGM) $(INPUT_SNPS) temp.gz
	time zcat temp.gz | sort -V -k1,1 -k2,2 | gzip > for_vep.temp.gz
	rm temp.gz
	mv for_vep.temp.gz for_vep.gz

	# Monitor vep progress in the background.
	./vep_monitor.sh &

	# NOTE: This step parallelizes well and takes a long time (hours on 1 core), consider bumping CPUs
	time perl $(VEP_PRGM) \
		-i for_vep.gz \
		-o vep_out.tmp \
		--tab \
		--dir_cache $(VEP_CACHE_DIR) \
		--cache \
		--force_overwrite \
		--compress_output gzip \
		--fork $(NPROC) \
		--buffer_size 100000 \
		--check_existing \
		--offline
	
	# Ignore any "Permission denied / syntax error" just before this it's just the monitor
	# script complaining that vep is closing the file it's monitoring a little bit before vep exits
	mv vep_out.tmp $(VEP_OUTPUT)


# Converts the vep output format to our expected output format.
$(VEP_SNP_OFILE): $(VEP_OUTPUT) $(UNI_CONVERT)
	time $(POST_VEP_PRGM) -u $(UNI_CONVERT) -o tmp_out.tsv -i $(VEP_OUTPUT)
	gzip tmp_out.tsv
	mv tmp_out.tsv.gz $(VEP_SNP_OFILE)


##### Final merged outputs ######
# Some score weights from opentargets for each of the VEP consequences.
VEP_CONSQ=vep_consequences.$(DUMA_GWAS_V2D_VER).tsv
$(VEP_CONSQ):
	curl -f https://raw.githubusercontent.com/opentargets/genetics-v2g-data/master/configs/vep_consequences.tsv > tmp.vep_cons.tsv
	mv tmp.vep_cons.tsv $(VEP_CONSQ)

# Merge the vep and otarget data into an output file.
$(OFILE): $(VEP_SNP_OFILE) $(OTARG_SNP_OFILE) $(VEP_CONSQ)
	./otarg_snp_merge.py \
		--orig-input $(VEP_SNP_OFILE) \
		--otarg-input $(OTARG_SNP_OFILE) \
		--vep-consequences $(VEP_CONSQ) \
		--output tmp.$(OFILE)

	# Copy the header over
	(zcat tmp.$(OFILE) || true) | head -1 | gzip > tmp.sorted.$(OFILE)
	# Sort the non-header data and write it in too
	zcat tmp.$(OFILE) | tail -n +2 | LC_ALL=C sort | gzip >> tmp.sorted.$(OFILE)	
	# Remove the tmp file and move over to output.
	rm tmp.$(OFILE)
	mv tmp.sorted.$(OFILE) $(OFILE)
	

# Convert the output file into an archive format for efficient lookup.
# Note we're calling directly into build_archive rather than make_zip.sh.
# make_zip doesn't support prefixes in its splitting yet, and we're presorted anyway.
# The 'tail' is to strip the header.
# If you want to change the prefix length, you'll have to save it somewhere as metadata and pass along to client.
$(OFILE_ZIP): $(OFILE)
	LEN=$$(zcat $(OFILE) | wc -l); \
		zcat $(OFILE) | tail -n +2 | ../duma_gwas/build_archive.py -t $$LEN -o $(OFILE_ZIP).tmp --prefix-length 6
	mv $(OFILE_ZIP).tmp $(OFILE_ZIP)

$(OFILE_D2G_SUMMARY): $(OFILE) $(INPUT_SNPS)
	time ./d2g_summary.py --v2g $(OFILE) --v2d $(INPUT_SNPS) --output-summary $(OFILE_D2G_SUMMARY)


##### clean ######

clean:
	rm -rf *.txt *.tsv *.tsv.gz temp tmp* *log rescue_vepd* for_vep raw_combined.gz vepd* *.tmp *.gz ensembl-vep/
