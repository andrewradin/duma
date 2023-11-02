default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

# The MeSH data is freely available for download from
# https://www.nlm.nih.gov/databases/download/mesh.html
# It consists of 3 ascii text files [cdq]YYYY.bin. The 'q' file seems to
# hold metadata terms. The 'c' file holds mostly compound records, which
# might be useful for drugname matching.
#
# The useful information seems to be in the 'd' file, which holds records
# that appear as nodes in the MeSH hierarchy (and have MN fields that hold
# their tree number).  Disease ('condition') records have a tree number
# starting with 'C'. This is the information that's extracted below.
#
# Field descriptions:
#   https://www.nlm.nih.gov/mesh/xmlconvert_ascii.html
#   https://www.nlm.nih.gov/mesh/dtype.html
# Categories & browser: https://meshb.nlm.nih.gov/treeView

FILE_CLASS=mesh

TMP=tempfile.tmp

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo MESH_VER=`date +%Y`
	@echo
	@echo 'Then:'
	@echo make input
	@echo make build
	@echo make publish_s3

DNLD=$(shell ../../web1/path_helper.py downloads)
DNLD_FILE=d$(MESH_VER).bin
DNLD_URL=ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/asciimesh
IFILE=$(DNLD)/$(FILE_CLASS).$(DNLD_FILE)

CAT_FILE=categories.tsv

OFILE=$(FILE_CLASS).$(OUT_VER).diseases.tsv
OUTPUTS=$(OFILE)

input: $(IFILE)
build: $(OUTPUTS)

$(IFILE):
	wget -O $(TMP) $(DNLD_URL)/$(DNLD_FILE)
	mv $(TMP) $(IFILE)

$(OFILE): $(IFILE) $(CAT_FILE)
	./parse_mesh.py --categories $(CAT_FILE) $(IFILE) $(OFILE)

clean:
	-rm *.tsv
	-rm $(TMP) $(TMP2)

