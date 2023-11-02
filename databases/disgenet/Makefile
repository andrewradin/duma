default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=disgenet

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)

SPECIES=HUMAN_9606
URL=https://www.disgenet.org/static/disgenet_ap1/files/downloads/
URL_FILE=curated_gene_disease_associations.tsv.gz
IFILE=disgenet.$(DISGENET_VER).curated_download.tsv.gz
IPATH=$(DNLD)$(IFILE)

OFILE=disgenet.$(OUT_VER).curated_meddra.tsv
OFILE_UMLS=disgenet.$(OUT_VER).curated_umls.tsv
DIS_FILE=disgenet.$(OUT_VER).disease_names.tsv
CUI_DIS_FILE=disgenet.$(OUT_VER).cui_disease_names.tsv

ARCHIVED_INPUTS=$(IPATH)

INPUTS=$(ARCHIVED_INPUTS)

OUTPUTS=\
	$(OFILE) \
	$(OFILE_UMLS) \
	$(DIS_FILE) \
	$(CUI_DIS_FILE) \
	# end of list

show_downloads:
	@echo $(INPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo DISGENET_VER=`date +%Y-%m-%d`
	@echo
	@echo 'Also ensure the latest/appropriate versions of'
	@echo 'Uniprot, UMLS, and MedDRA are listed'
	@echo 'For MedDRA the version needs to be aligned with what is used in UMLS.'
	@echo 'See note in versions.py in this director for more info'
	@echo
	@echo 'Then:'
	@echo make download_input
	@echo make build
	@echo make publish_s3


input: $(INPUTS)
build: $(OUTPUTS)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt

# This allows a manual download from the website to the current versioned
# filename. Only use if you're creating a new input version.
download_input:
	curl $(URL)$(URL_FILE) > $(IPATH)

$(OUTPUTS): $(IPATH)
	zcat $(IPATH) \
		| ./parse_disgenet.py \
		$(SPECIES).$(UNIPROT_VER) $(MEDDRA_VER) $(UMLS_VER) \
		temp1 temp2 temp3 > tmp
	mv tmp $(OFILE)
	mv temp1 $(DIS_FILE)
	mv temp2 $(CUI_DIS_FILE)
	mv temp3 $(OFILE_UMLS)

clean:
	rm -rf *.tsv *.txt *.log
