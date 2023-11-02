default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=uniprot_dpi

OFILE=$(FILE_CLASS).$(OUT_VER).tsv
OUTPUTS=$(OFILE)

WS=$(shell ../../web1/path_helper.py storage)
UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
UNIPROT_CONVERTER=$(WS)$(UNI_BUCKET)/$(UNI_FILE)

INPUTS=\
	$(UNIPROT_CONVERTER) \
    # end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '====================================='
	@echo
	@echo 'Ensure the latest version of Uniprot is set'
	@echo
	@echo 'make input'
	@echo 'make build'
	@echo 'make publish_s3'
	@echo

clean:
	-rm *.tsv

$(OFILE): $(INPUTS)
	./build_dpi.sh $(UNIPROT_CONVERTER) > tmp.tsv
	mv tmp.tsv $(OFILE)

