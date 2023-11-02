default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=finngen
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

# Download the manifest linked off of https://finngen.gitbook.io/documentation/data-download
# and then filtered by Aaron.
MANIFEST_NAME=finngen_whitelist_$(FINNGEN_INPUT_VER).tsv
S3_FINNGEN_MANIFEST=s3://duma-datasets/$(MANIFEST_NAME)

DNLD=$(shell ../../web1/path_helper.py downloads)
MANIFEST=$(DNLD)$(MANIFEST_NAME)

# hand annotated
RAW_DNLD=$(DNLD)$(OFILE_STEM)/
RAW_DNLD_TMP=$(DNLD)$(OFILE_STEM).tmp/

OSNPS=$(OFILE_STEM).data_filtered.tsv.gz
OSTUDIES=$(OFILE_STEM).studies.tsv

ARCHIVED_INPUTS=$(MANIFEST)

show_downloads:
	@echo $(INPUTS)

show_latest_version:
	@echo
	@echo '========================================='
	@echo 'Visit https://finngen.gitbook.io/documentation/data-download and see if there is a newer release than we already have.'
	@echo
	@echo 'No need to generate a new version if it is not a newer release'
	@echo
	@echo 'If there is a new version, download the new manifest, filter by hand and place at $(MANIFEST_NAME).'
	@echo
	@echo '(NOTE: You probably want a big machine at this point e.g. 12xlarge takes ~40 minutes)'
	@echo
	@echo 'Update the version in versions, then run:'
	@echo make input
	@echo make build
	@echo make publish_s3

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt


INPUTS=\
	$(RAW_DNLD) \
	# end of list

OUTPUTS=\
	$(OSNPS) \
	$(OSTUDIES) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

$(INPUTS): $(ARCHIVED_INPUTS)
	rm -rf $(RAW_DNLD_TMP)
	mkdir -p $(RAW_DNLD_TMP)
	# Similar to UKBB there's too much raw data to keep it all, pre-filter at some p-value
	# This is mostly CPU-bound, you should probably run this on a big machine.
	# Takes ~40 minutes on a 12xlarge.
	./download_snps.py -i $(MANIFEST) -o $(RAW_DNLD_TMP) -p 0.01
	mv $(RAW_DNLD_TMP) $(RAW_DNLD)

$(OSNPS) $(OSTUDIES): $(INPUTS)
	./parse_snps.py  \
		--output-data $(OSNPS) \
		--output-studies $(OSTUDIES) \
		--input-dir $(RAW_DNLD)/ \
		--input-manifest $(MANIFEST) \
		--p-primary  1e-6 \
		--p-secondary 1e-2 \
		--p-secondary-prots 100

clean:
	rm -rf *.gz *.tsv *.tmp
