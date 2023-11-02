default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=monarch

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)

URL_BASE=https://data.monarchinitiative.org/latest/tsv/

DIS_IFILE_BASE=disease_phenotype.all.tsv
DIS_IFILE=$(DIS_IFILE_BASE).gz
DIS_IURL=$(URL_BASE)disease_associations/$(DIS_IFILE)
V_DIS_IFILE=$(DNLD)$(FILE_CLASS).$(MONARCH_VER).$(DIS_IFILE)


# For now we're just using human gene-phenotypes
# This was done largely for ease and
# so we can continue to say we're using exclusively human-derived data
#
# I do have questions about the translation of animal model phenotypes, but
# we could imagine using animal models, though
# if we do this is the file we'd want.
# The licensces for non-human DBs have been assessed already
# We would still need to figure out how to map phenotypes b/t species
# in addition to the more straightforward gene mapping
#PHENO_IFILE_BASE=gene_phenotype.all.tsv


GEN_IFILE_BASE=gene_phenotype.9606.tsv
GEN_IFILE=$(GEN_IFILE_BASE).gz
GEN_IURL=$(URL_BASE)gene_associations/$(GEN_IFILE)
V_GEN_IFILE=$(DNLD)$(FILE_CLASS).$(MONARCH_VER).$(GEN_IFILE)

#TODO: everything under here

DIS_OFILE=$(FILE_CLASS).$(OUT_VER).disease.tsv
EVID_OFILE=$(FILE_CLASS).$(OUT_VER).evidence.tsv
GEN_OFILE=$(FILE_CLASS).$(OUT_VER).gene.tsv

ARCHIVED_INPUTS= \
	$(V_GEN_IFILE) \
	$(V_DIS_IFILE) \
	# end of list

INPUTS=$(ARCHIVED_INPUTS)

OUTPUTS=\
	$(DIS_OFILE) \
	$(EVID_OFILE) \
	$(GEN_OFILE) \
	# end of list

show_downloads:
	@echo $(INPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo MONARCH_VER=`date +%Y-%m-%d`
	@echo
	@echo 'Also ensure the latest/appropriate versions of'
	@echo 'Uniprot is listed'
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
	curl $(DIS_IURL) > tmp
	mv tmp $(V_DIS_IFILE)
	curl $(GEN_IURL) > tmp
	mv tmp $(V_GEN_IFILE)

$(DIS_OFILE): $(V_DIS_IFILE)
	./parse_monarch_disease.py \
		$(V_DIS_IFILE) \
		temp1
	mv temp1 $(DIS_OFILE)

$(EVID_OFILE): $(V_DIS_IFILE) $(V_GEN_IFILE)
	./parse_monarch_evidence.py \
		$(V_DIS_IFILE) $(V_GEN_IFILE) \
		temp1
	mv temp1 $(EVID_OFILE)

$(GEN_OFILE): $(V_GEN_IFILE)
	./parse_monarch_gene.py \
		$(V_GEN_IFILE) $(UNIPROT_VER) \
		temp1
	mv temp1 $(GEN_OFILE)

clean:
	rm -rf *.tsv *.txt *.log
