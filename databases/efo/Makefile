default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=efo

include ../version_tools/setup.mk
include versions.mk

PRGM=parse_efo.py

IURL=https://www.ebi.ac.uk/efo/efo.obo
IFILE=efo.$(EFO_VER).obo

# EFO has a special OpenTargets version of its ontology that includes all of the opentargets
# specific terms.  It's only available in OWL format right now.
IURL_OTARG=https://github.com/EBISPOT/efo/releases/download/current/efo_otar_profile.owl
IFILE_OTARG=efo.$(EFO_VER).otarg.owl

IDIR=$(shell ../../web1/path_helper.py downloads)
INPUT=$(IDIR)$(IFILE)
INPUT_OTARG=$(IDIR)$(IFILE_OTARG)
ARCHIVED_INPUTS=$(INPUT)

OFILE_NAME=efo.$(OUT_VER).obo.pickle
OFILE_ALL_TERMS=efo.$(OUT_VER).terms.tsv
OFILE_OTARG_HIER=efo.$(OUT_VER).otarg_hier.sqlsv
OUTPUTS=\
	$(OFILE_NAME) \
	$(OFILE_ALL_TERMS) \
	$(OFILE_OTARG_HIER) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(INPUT) $(INPUT_OTARG)
build: $(OUTPUTS)

$(INPUT) $(INPUT_OTARG):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_download

show_latest_version:
	./show_latest_version.py $(IURL)

# This allows a manual download from the website to the current versioned
# filename. Only use if you're creating a new input version.
download_input:
	curl $(IURL) > temp1
	# verify version before renaming
	head temp1 | sed -e 's/\./_/g' | grep -q $(EFO_VER)
	mv temp1 $(INPUT)

	curl -Lf $(IURL_OTARG) > temp2
	mv temp2 $(INPUT_OTARG)

$(OFILE_NAME) $(OFILE_ALL_TERMS): $(INPUT)
	python $(PRGM) $(INPUT) temp1 temp2
	mv temp1 $(OFILE_NAME)
	mv temp2 $(OFILE_ALL_TERMS)

$(OFILE_OTARG_HIER): $(INPUT_OTARG)
	./parse_efo_otar.py -i $(INPUT_OTARG) -o $(OFILE_OTARG_HIER)

clean:
	-rm *.tsv *.pickle
