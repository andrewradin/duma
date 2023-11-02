default:
	@echo "USAGE: make input|build|publish_ws|publish_s3|clean"

GLEE_DIR=$(shell ../../web1/path_helper.py glee)
OUTDIR=./outputs
STOR=$(shell ../../web1/path_helper.py storage)
CNVRT_PRGM=./parse_msigdb_entrez.py

V=v6.0
MSIGDB_ZIP_DIR=msigdb_$(V)_files_to_download_locally
MSIGDB_ZIP=$(MSIGDB_ZIP_DIR).zip
MSIGDB_INR_DIR=msigdb_$(V)_GMTs
MSIGDB_DIR=$(MSIGDB_ZIP_DIR)/$(MSIGDB_INR_DIR)
IDIR=$(shell ../../web1/path_helper.py downloads)

INPUT=$(IDIR)$(MSIGDB_ZIP)
FULL_IDIR=$(IDIR)$(MSIGDB_DIR)/

ENTREZ_FILENAME=HUMAN_9606_Uniprot_data.tsv
ENTREZ=$(STOR)$(ENTREZ_FILENAME)

OFILE_SUF=.uniprot.gmt

OUTPUTS=\
	$(OUTDIR) \
	# end of list

BUCKET=s3://2xar-duma-glee/

show_downloads:
	@echo $(INPUT)
	@echo $(FULL_IDIR)

input: $(FULL_IDIR) $(ENTREZ)
build: $(OUTPUTS)

$(INPUT):
	s3cmd get $(BUCKET)$(MSIGDB_ZIP) $(INPUT)
$(FULL_IDIR): $(INPUT)
	unzip -D $(INPUT) -d $(IDIR)

$(ENTREZ):
	s3cmd get s3://duma-datasets/$(ENTREZ_FILENAME) $(ENTREZ)

$(OUTPUTS): $(FULL_IDIR) $(ENTREZ)
	rm -rf $(OUTDIR)
	$(CNVRT_PRGM) $(FULL_IDIR) $(ENTREZ) $(OUTDIR)

publish_ws: $(OUTPUTS)
	cp $(OUTDIR)/* $(GLEE_DIR)

publish_s3: publish_ws
	@for FILE in $(GLEE_DIR)/*$(OFILE_SUF);\
	do \
		echo writing $$FILE...;\
		s3cmd put $$FILE $(BUCKET);\
	done

clean:
	-rm -rf $(OUTDIR) $(IDIR)$(MSIGDB_ZIP_DIR)
