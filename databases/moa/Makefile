default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=moa
include ../version_tools/setup.mk
include versions.mk


INPUTS=../matching/moa_attrs.v$(MATCHING_VER).tsv
OUTPUTS=moa.full.$(OUT_VER).attributes.tsv

show_latest_versions:
	@echo
	@echo '============================================='
	@echo 'This should run after 'make build' in /matching'
	@echo 'It just renames an output file from there and publishes it.'
	@echo 'So you only need to run make build and publish_s3'

build: $(OUTPUTS)

$(OUTPUTS): $(INPUTS)
	cp -v $(INPUTS) $(OUTPUTS)

clean:
	-rm *.tsv
