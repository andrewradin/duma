default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=orthologs

include ../version_tools/setup.mk
include versions.mk

WS=$(shell ../../web1/path_helper.py storage)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)

# We're pulling orthology data from Ensembl BioMART.
# This gives similar, but not identical results to https://web.expasy.org/
# For the most part, the similarity numbers are very close, but there are a few cases
# where data seems to be missing.
# e.g. BioMART misses zebrafish_CASPA (CASP1), Expasy misses RAT_JAK1.

ORTHO_IN=orthology_in.$(OUT_VER).tsv
# You can generate this file by visiting http://uswest.ensembl.org/biomart, picking
# the attributes / orthologs you care about in the menus, and then clicking the "XML" button.
# Then add "query=" to the start of it.
# Be sure to update parse.py as well to handle whatever columns you added/removed.
QUERY_FILE=query.xml
ORTHO_OUT=$(FILE_CLASS).$(OUT_VER).sqlsv

UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
ENSEMBL_UNIPROT_CONVERTER=$(WS)$(UNI_BUCKET)/$(UNI_FILE)

# Human, mouse, rat, dog, zebrafish
# STRING doesn't have 9615 for dog, 9612 is the closest.
ANIMALS=9606 10090 10116 9612 7955
STRINGV=$(STRINGDB_VERSION)
STRINGVd=$(subst .,-,$(STRINGV))
STRING_URL=https://version-$(STRINGVd).string-db.org/download/protein.links.v$(STRINGV)/
STRING_FILE_NAME_SUFFIX=protein.links.v$(STRINGV).txt.gz
STRING_ANI_FILES=$(foreach ANI, $(ANIMALS), $(ANI).$(STRING_FILE_NAME_SUFFIX))

OUTPUTS=\
	$(ORTHO_OUT) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(ORTHO_IN) $(STRING_ANI_FILES)

build: $(OUTPUTS)


$(STRING_ANI_FILES):
	curl -fL $(STRING_URL)$@ > tmp
	mv tmp $@

$(ORTHO_IN):
	# Generate the header
	cat query.xml | grep "Attribute" | cut -d'"' -f2 | tr '\n' '\t' > tmp.tmp
	# Pull down the rest of the data
	curl -d @query.xml http://uswest.ensembl.org/biomart/martservice >> tmp.tmp

	# Move it into place.
	mv tmp.tmp $(ORTHO_IN)

$(ORTHO_OUT): $(ORTHO_IN) $(ENSEMBL_UNIPROT_CONVERTER) $(STRING_ANI_FILES)
	./parse.py -i $(ORTHO_IN) -o $(ORTHO_OUT) -u $(ENSEMBL_UNIPROT_CONVERTER) $(STRING_ANI_FILES)


show_latest_version:
	@echo
	@echo '======================================'
	@echo "The underlying data here likely doesn't change much - but it does depend on the uniprot converter"
	@echo "This also makes use of STRING data outside of our ETL and thus you'll want to verify the latest"
	@echo "full version here: https://string-db.org/cgi/download"
	@echo
	@echo 'make input'
	@echo 'make build'
	@echo 'make publish_s3'

clean:
	-rm *.tsv *.tmp

