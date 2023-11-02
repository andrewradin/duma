default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=mondo

include ../version_tools/setup.mk
include versions.mk

PRGM=parse_efo.py

IURL=http://purl.obolibrary.org/obo/mondo/mondo-with-equivalents.json
IFILE=mondo.$(OUT_VER).json
IDIR=$(shell ../../web1/path_helper.py downloads)
INPUT=$(IDIR)$(IFILE)
ARCHIVED_INPUTS=$(INPUT)

OFILE_MAPPINGS=mondo.$(OUT_VER).mappings.tsv
OFILE_LABELS=mondo.$(OUT_VER).labels.tsv
OUTPUTS=\
	$(OFILE_MAPPINGS) \
	$(OFILE_LABELS) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUT)
build: $(OUTPUTS)

$(INPUT):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_download

show_latest_version:
	@echo
	@echo '==============================================='
	@echo '- Update versions.py to add a new version.'
	@echo '- make build will download the latest source and process it.'
	@echo '- make publish_s3 will both publish the new version and archive the input.'


# This allows a manual download from the website to the current versioned
# filename. Only use if you're creating a new input version.
download_input:
	curl -L $(IURL) > temp1
	mv temp1 $(INPUT)

$(OUTPUTS): $(INPUT)
	./parse_mondo.py --mondo-data $(INPUT) --out-mappings $(OFILE_MAPPINGS) --out-labels $(OFILE_LABELS)

clean:
	-rm *.tsv
