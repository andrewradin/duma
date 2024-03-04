default:
	@echo "USAGE: make input|build|clean"

.PHONY: input check-abbrv
check-abbrv:
ifndef ABBRV
    $(error ABBRV is undefined)
endif

WD=$(shell pwd)
DWNLD_PRGM=./firehose_get

R_DIR=$(shell ../../web1/path_helper.py Rscripts)
SIG_PRGM=TCGA_sig.R

IDIR=$(shell ../../web1/path_helper.py downloads)
ODIR=$(shell ../../web1/path_helper.py storage)

DATE=2016_01_28
empty:=
CMPCT_DATE=$(strip $(subst _,$(empty), $(DATE)))

DWNLD_DIR=stddata__$(DATE)/$(ABBRV)/$(CMPCT_DATE)

SHARED=rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data
PREF=gdac.broadinstitute.org_$(ABBRV).Merge_$(SHARED).Level_3.$(CMPCT_DATE)00.0.0
TARFILE=$(PREF).tar.gz
RAWFILE=$(PREF)/$(ABBRV).$(SHARED).data.txt
DATA_FILE_PREFIX=$(ABBRV)_$(DATE)
DATA_FILE=$(DATA_FILE_PREFIX).tsv.gz

INPUTS=$(DATA_FILE)
OUTPUT=$(DATA_FILE_PREFIX)_sigProtTable.tsv

S3_BUCKET=s3://2xar-duma-sigprot

input: $(INPUTS)
build: $(OUTPUT)

$(DATA_FILE): $(DWNLD_PRGM) check-abbrv
	$(DWNLD_PRGM) -b -tasks genes_normalized stddata $(DATE) $(ABBRV)
	mv $(DWNLD_DIR)/* $(IDIR)
	tar -xzvf $(IDIR)$/$(TARFILE)
	gzip -c $(RAWFILE) > temp
	mv temp $(DATA_FILE)

$(DWNLD_PRGM):
	wget http://gdac.broadinstitute.org/runs/code/firehose_get_latest.zip
	unzip firehose_get_latest.zip

$(OUTPUT): $(DATA_FILE) check-abbrv
	cd $(R_DIR) && Rscript $(SIG_PRGM) $(WD) $(DATA_FILE)
	mv browse_significantprotein.tsv $(OUTPUT)

publish: $(OUTPUT) check-abbrv
	s3cmd put $(OUTPUT) $(S3_BUCKET)

clean:
	rm -rf *.tsv firehose_get_latest.zip gdac* stddata* *png *gz
