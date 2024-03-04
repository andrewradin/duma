default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=selleckchem
include ../version_tools/setup.mk
include versions.mk

WS_DNLD=$(shell ../../web1/path_helper.py downloads)
TMP_IN=in.tmp
TMP_OUT=out.tmp

# See top of versions.py for how to set up a new version entry.
#
# Although the filenames contain date stamps, edits are made without
# changing the datestamps. We could archive these, but in order to
# do that and still be able to pick up improvements we'd need to
# layer an entirely new scheme on top of theirs. Instead, we'll just
# live with possible inconsistency, since this data isn't critical.
INPUTS=$(patsubst %,$(WS_DNLD)selleckchem_%,$(SELLECKCHEM_FILES))

ATTR_FILE=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv

# quotes are needed because some filenames include parentheses
$(WS_DNLD)selleckchem_%:
	wget -O $(TMP_OUT) 'https://file.selleckchem.com/downloads/library/$*'
	mv $(TMP_OUT) '$@'

OUTPUTS=\
	$(ATTR_FILE) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	./show_latest_version.py

$(ATTR_FILE): $(INPUTS)
	./parse_schem.py -o $(TMP_OUT) $(patsubst %,'%',$(INPUTS))
	mv $(TMP_OUT) $(TMP_IN)
	../matching/make_std_smiles.py -r worker-test -i $(TMP_IN) -o $(TMP_OUT)
	mv $(TMP_OUT) $(ATTR_FILE)

clean:
	-rm *.tsv

