default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=lincs

WS=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

# This link and the files are listed out here: https://clue.io/data/CMap2020#LINCS2020
# However the links aren't plain links, you either have to click and start the download
# or inspect source to get the underlying data URLs.
BASE_URL=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020
DNLD_DIR=$(DNLD)lincs/$(LINCS_VER)

# The main data matrix.  This contains the treatment compounds, there are other versions
# available with e.g. crispr, or controls.
TREAT_DATA=$(DNLD_DIR)/level5_beta_trt_cp_n720216x12328.gctx
# Metadata files with more information about the cells, compounds, etc.
CELL_INFO=$(DNLD_DIR)/cellinfo_beta.txt
COMPOUND_INFO=$(DNLD_DIR)/compoundinfo_beta.txt
SIG_INFO=$(DNLD_DIR)/siginfo_beta.txt
GENE_INFO=$(DNLD_DIR)/geneinfo_beta.txt


OFILE_STEM=$(FILE_CLASS).full.$(OUT_VER)
OFILE_EXPR=$(OFILE_STEM).expression.npz
OFILE_META=$(OFILE_STEM).metadata.json
OFILE_ATTR=$(OFILE_STEM).attributes.tsv
OFILE_DPI=$(OFILE_STEM).evidence.tsv
TMP_ATTR_PRESTD=$(OFILE_STEM).prestd.tsv


INPUTS=\
	$(TREAT_DATA) \
	$(CELL_INFO) \
	$(COMPOUND_INFO) \
	$(SIG_INFO) \
	$(GENE_INFO)
    # end of list

OUTPUTS=\
	$(OFILE_EXPR) \
	$(OFILE_META) \
	$(OFILE_ATTR) \
	$(OFILE_DPI) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '========================================='
	@echo
	@echo 'Update the Uniprot version to the latest'
	@echo 'Check the following site to see if there is a new LINCS download:'
	@echo ' https://clue.io/data/CMap2020#LINCS2020'
	@echo "Though note this is the first version so we're not sure what the update policy is"
	@echo "It looks like they may just be updating in place and not changing file names."
	@echo "If so, we'd want to archive inputs - i.e. there will likely be ETL updates needed"
	@echo
	@echo 'Then run:'
	@echo make input
	@echo "make build - note this will use the worker-test machine for some processing"
	@echo make publish_s3

$(CELL_INFO) $(COMPOUND_INFO) $(SIG_INFO) $(GENE_INFO):
	mkdir -p $(DNLD_DIR)
	curl --fail $(BASE_URL)/$(notdir $@) > $@.tmp
	mv $@.tmp $@

$(TREAT_DATA):
	mkdir -p $(DNLD_DIR)
	curl --fail $(BASE_URL)/level5/$(notdir $@) > $@.tmp
	mv $@.tmp $@


$(TMP_ATTR_PRESTD) $(OFILE_DPI): $(COMPOUND_INFO)
	./make_drugset.py -i $(COMPOUND_INFO) --out-attr $(TMP_ATTR_PRESTD) --out-dpi $(OFILE_DPI) -u $(UNIPROT_VER)

$(OFILE_ATTR): $(TMP_ATTR_PRESTD)
	../matching/make_std_smiles.py -i $(TMP_ATTR_PRESTD) -o $(OFILE_ATTR) -r worker-test

$(OFILE_EXPR) $(OFILE_META): $(INPUTS) $(OFILE_ATTR)
	./parse_sigs.py \
		--input-data-dir $(DNLD_DIR) \
		--input-attrs $(OFILE_ATTR) \
		--out-expr $(OFILE_EXPR) \
		--out-meta $(OFILE_META) \
		-u $(UNIPROT_VER)

clean:
	rm -rf *.tsv *.npz
