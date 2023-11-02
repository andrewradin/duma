default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=ucsc_hg

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)

CHRM_PARSE_PRGM=./parse_hg_chrms.py
GENE_GET_PRGM=./get_gene_lengths.sh
GENE_PARSE_PRGM=./parse_hg_gene.py

CHRM_URL=https://genome.ucsc.edu/goldenpath/help/
CHRM_URLFILE=hg$(HG_VER).chrom.sizes
CHRM_IFILE=$(FILE_CLASS).$(HG_VER).chrom.sizes
CHRM_INPUT=$(DNLD)$(CHRM_IFILE)

GENE_IFILE=$(FILE_CLASS).$(HG_VER).gene.sizes
GENE_INPUT=$(DNLD)$(GENE_IFILE)

INPUTS=\
	$(CHRM_INPUT) \
	$(GENE_INPUT) \
	# end of list

$(CHRM_INPUT):
	curl $(CHRM_URL)$(CHRM_URLFILE) > temp
	mv temp $(CHRM_INPUT)

$(GENE_INPUT):
	$(GENE_GET_PRGM) $(HG_VER) > temp
	mv temp $(GENE_INPUT)

CHRM_OFILE=$(FILE_CLASS).$(OUT_VER).chrom_sizes.tsv
GENE_OFILE=$(FILE_CLASS).$(OUT_VER).prot_txn_sizes.tsv

OUTPUTS=\
	$(CHRM_OFILE) \
	$(GENE_OFILE) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo "Check UCSC to see if a new HG version has been released."
	@echo "If so update versions.py, then run:"
	@echo "make input"
	@echo "make build"
	@echo "make publish_s3"

$(CHRM_OFILE): $(CHRM_INPUT)
	$(CHRM_PARSE_PRGM) $(CHRM_INPUT) > temp
	mv temp $(CHRM_OFILE)

$(GENE_OFILE): $(GENE_INPUT)
	$(GENE_PARSE_PRGM) $(GENE_INPUT) > temp
	mv temp $(GENE_OFILE)

clean:
	rm -rf *.tsv *.sizes temp
