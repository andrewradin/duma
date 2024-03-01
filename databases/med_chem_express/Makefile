default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=med_chem_express

include ../version_tools/setup.mk
include versions.mk

COLLECTION=med_chem_express

TMP_IN=in.tmp
TMP_OUT=out.tmp

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

INPUT=$(WS_DNLD)med_chem_express_$(MED_CHEM_EXPRESS_VER).xlsx
ARCHIVED_INPUTS=$(INPUT)

ATTR_FILE=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv

OUTPUTS=\
	$(ATTR_FILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUT)
build: $(OUTPUTS)


show_latest_version:
	@echo -run:
	@echo make download_input
	@echo
	@echo 'If the file is sufficiently different update versions.py'
	@echo with MED_CHEM_EXPRESS_VER=`date +'%Y-%m-%d'`
	@echo
	@echo '- Then run:'
	@echo make build
	@echo make publish_s3

# There seems to be a single URL without any particular versioning.
# So, manually run a download step to grab the file and timestamp it,
# and if it's sufficiently different, add it to versions.py
download_input:
	wget -O $(TMP_OUT) 'http://file.medchemexpress.com/mce_compoundDetail!downloadSingle.shtml?downloadSinglePath=HY-L001P&downloadSingleName=HY-L001P.xlsx'
	mv $(TMP_OUT) $(WS_DNLD)$(FILE_CLASS)_`date +'%Y-%m-%d'`.xlsx

$(ATTR_FILE): $(INPUT)
	./parse_mce.py $(INPUT) -o $(TMP_OUT)
	mv $(TMP_OUT) $@

clean:
	-rm *.tsv

