default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=phewas

PRS_PRGM=./parse_phewas.py
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

DNLD=$(shell ../../web1/path_helper.py downloads)
ODIR=$(shell ../../web1/path_helper.py storage)
S3_STORAGE=s3://duma-datasets/

URL_PREF=https://phewascatalog.org/files/
IFILE=phewas-catalog
CSV=.csv
ZIP=.zip
DAT_INPUT=$(DNLD)$(FILE_CLASS).$(PHEWAS_DNLD_VER).$(IFILE)$(CSV)
TMP_DNLD=$(DNLD)tmp
CHAIN_FILE=$(ODIR)hg18ToHg38.over.chain.gz

OUT1=$(OFILE_STEM).data.tsv.gz
OUT2=$(OFILE_STEM).studies.tsv

INPUTS=\
	$(DAT_INPUT) \
	$(CHAIN_FILE) \
	# end of list

OUTPUTS=\
	$(OUT1) \
	$(OUT2) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Go to https://phewascatalog.org/files/'
	@echo 'And look for phewas-catalog.csv and check the upload date.'
	@echo 'Compare that to PHEWAS_DNLD_VER in versions.py'
	@echo 'Update if they differ and run:'
	@echo make input
	@echo make build
	@echo make publish_s3

$(CHAIN_FILE):
	../CrossMap/get_chain_files.py $(CHAIN)

$(DAT_INPUT):
	curl $(URL_PREF)$(IFILE)$(CSV)$(ZIP) > $(TMP_DNLD)$(ZIP)
	unzip $(TMP_DNLD)$(ZIP)
	rm -rf __MACOSX/
	mv $(IFILE)$(CSV) $(DAT_INPUT)

$(OUTPUTS): $(INPUTS)
	$(PRS_PRGM) -dataset $(DAT_INPUT) -output1 tmp1 -output2 tmp2 --chain_file $(CHAIN_FILE)
	gzip -c tmp1 > $(OUT1)
	rm tmp1
	mv tmp2 $(OUT2)

clean:
	rm -rf *.tsv *.txt *.log *gz tmp*
