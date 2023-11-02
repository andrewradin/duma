default:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

FILE_CLASS=faers

include ../version_tools/setup.mk
include versions.mk

# FAERS is a bit more complex than most ETL. The steps are:
# - check the web for new files, and update the URL_DIR if neccesary
# - download and unpack any files listed in URL_DIR that aren't in DNLDF
# - accumulate everything in DNLDF into a single set of files
# - map names and deduplicate accumulated data
# - package data in matrix files
#
# 'make input' does the first 3 steps, and 'make build' does the last 2.
#
# The FDA does occasional inconsistent things that may cause the accumulate
# process to die. For example, the demographics file for 2018Q1 was named
# demo18q1_new.txt rather than demo18q1.txt. In that case, and maybe in
# future cases, the easiest fix was to just rename the file in the extracted
# directory and re-run the 'make input' step.
#
# LAST_INPUT is set from FAERS_VER in versions.py, and indicates the last
# chunk of FAERS data expected in this version. It is used as part of the names
# of the accumulation files (INPUTS) and to set LAST_INPUT_FLAG.
#
# LAST_INPUT_FLAG is a directory that will exist if the latest FAERS
# dataset to be included has been downloaded successfully. If it doesn't
# exist, it triggers a makestep that updates URL_DIR with the urls of
# all available downloads, and then downloads any datasets not present.
#
# All other intermediate files include the final FAERS output version
# in their names. This lets us skip the (somewhat slow) building of the
# accumulation files if we're generating a new version due to mapping
# or algorithm changes, rather than to include additional FAERS data.
#
# It seems reasonable to think of FAERS and CVAROD as different flavors
# of the same FILE_CLASS. But since they're actually separate data sources,
# they could be versioned independently, and require some Makefile and
# processing customizations. So, each will continue to be managed in a
# separate directory, and the OUTPUTS are named accordingly.

URL_DIR = urls
DNLDF=$(shell ../../web1/path_helper.py downloads)faers
LAST_INPUT=$(FAERS_VER)
LAST_INPUT_DIR=$(DNLDF)/faers_ascii_$(LAST_INPUT)
# include the ascii subdir so this target doesn't conflict with the
# target doing the actual download/extract
LAST_INPUT_FLAG=$(LAST_INPUT_DIR)/ascii

INPUT_TYPES=drug indi demo date
INPUTS=$(foreach type,$(INPUT_TYPES),raw.$(LAST_INPUT).$(type).tsv)
USAGE_INPUT=raw.$(LAST_INPUT).usage.tsv
DEDUPS=$(foreach type,$(INPUT_TYPES),dedup.$(OUT_VER).$(type).tsv)

N2CDIR=$(shell ../../web1/path_helper.py storage)name2cas
N2CFILE=$(N2CDIR)/name2cas.$(NAME2CAS_VER).tsv

CASE_MAP=casemap.$(LAST_INPUT).pkl

MAT_STEM=$(FILE_CLASS).$(OUT_VER)
STANDARD_OUTPUTS=\
	$(MAT_STEM).drug_mat.npz \
	$(MAT_STEM).drug_cols.txt \
	$(MAT_STEM).indi_mat.npz \
	$(MAT_STEM).indi_cols.txt \
	$(MAT_STEM).demo_mat.npz \
	$(MAT_STEM).date_mat.npz \
	# end of list

USAGE_OUTPUTS=\
	$(MAT_STEM).indi_drug_mat.npz \
	$(MAT_STEM).dose_mat.npz \
	$(MAT_STEM).indi_drug_dose_meta.json \
	# end of list

OUTPUTS=$(STANDARD_OUTPUTS) $(USAGE_OUTPUTS)

show_latest_version:
	@echo
	@echo 'To find the latest FAERS version go to:'
	@echo 'https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html'
	@echo 'Then update in versions.py accordingly'
	@echo
	@echo 'Then proceed as normal by running:'
	@echo 'make input'
	@echo '  NOTE: the FDA can be inconsistent in file naming and this can'
	@echo '        casuse input to fail. If this happens see note in Makefile.'
	@echo 'make build'
	@echo 'make publish_s3'


input: $(INPUTS) $(USAGE_INPUT)

build: $(OUTPUTS)

# This happens when the last expected input doesn't exist.
# Make sure URL list is up to date, and then download any new files.
$(LAST_INPUT_FLAG):
	./fetchFAERSData.py
	$(MAKE) download

$(CASE_MAP): $(LAST_INPUT_FLAG)
	./make_case_map.py $(DNLDF) faers_ascii_$(LAST_INPUT) $@

$(INPUTS): $(LAST_INPUT_FLAG) $(CASE_MAP)
	./accumulate_raw.py $(DNLDF) faers_ascii_$(LAST_INPUT) --case-map $(CASE_MAP) $@

$(USAGE_INPUT): $(LAST_INPUT_FLAG) $(CASE_MAP)
	./accumulate_usage.py $(DNLDF) faers_ascii_$(LAST_INPUT) --case-map $(CASE_MAP) $@

$(N2CDIR)/%:
	$(S3_TOOL) name2cas $*

mapped.$(OUT_VER).drug.tsv:raw.$(LAST_INPUT).drug.tsv $(N2CFILE)
	./map_drugs.py $(N2CFILE) $< $@

mapped.$(OUT_VER).indi.tsv:raw.$(LAST_INPUT).indi.tsv
	./map_indis.py $(MEDDRA_VER) $< $@

mapped.$(OUT_VER).usage.tsv: $(USAGE_INPUT)
	./map_usage.py --name2cas $(N2CFILE) --meddra-ver $(MEDDRA_VER) $< $@

dedup.%.tsv:mapped.%.tsv
	./deduplicate.py $< $@

dedup.$(OUT_VER).demo.tsv:raw.$(LAST_INPUT).demo.tsv
	./demo_deduplicate.py $< $@

dedup.$(OUT_VER).date.tsv:raw.$(LAST_INPUT).date.tsv
	./date_deduplicate.py $< $@

$(USAGE_OUTPUTS): mapped.$(OUT_VER).usage.tsv
	./usage_dedupe_and_matrix.py mapped.$(OUT_VER).usage.tsv $(MAT_STEM).

$(STANDARD_OUTPUTS):$(DEDUPS)
	./make_matrix_files.py dedup.$(OUT_VER) $(MAT_STEM)


TMP_DIR = tmp
NEW_DOWNLOADS = new_downloads
TMP_SD = $(TMP_DIR)/tmp
ALL_URLS=$(wildcard $(URL_DIR)/*)
ALL_TARGETS=$(patsubst $(URL_DIR)/%,$(DNLDF)/faers_ascii_%,$(ALL_URLS))

$(DNLDF)/faers_ascii_%: $(URL_DIR)/%
	#Always start with fresh tmp directory
	rm -rf $(TMP_DIR)
	mkdir $(TMP_DIR)
	wget -P $(TMP_DIR) `cat $(URL_DIR)/$*`
	unzip -o $(TMP_DIR)/*.zip -d $(TMP_SD) 
	echo "downloading and unzipping complete"
	# Variable holds the original ASCII directory name which is standardized across all files to ascii
	./processFAERS.sh $(TMP_SD)
	test -s $(TMP_DIR)/complete 
	mkdir -p $@
	mv $(TMP_SD)/ascii $@
	rm -rf $(TMP_DIR)
	touch $(NEW_DOWNLOADS)

download: $(ALL_TARGETS)

# this is a backup step for the initial creation of the new_downloads file;
# it will try to get any missing data, but then force the file to exist
# even if nothing new gets downloaded
$(NEW_DOWNLOADS):
	$(MAKE) check_fda download
	touch $(NEW_DOWNLOADS)

clean:
	rm -rf $(TMP_DIR) *.txt *.tmp
