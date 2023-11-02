default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=homologene

include ../version_tools/setup.mk
include versions.mk

PARSE_PRGM=parse_homologene.py
XML_BUILD_PRGM=build_xml_queries.py
CNVRT_PRGM=convert_entrez_to_ensemblTRS.py

URL=ftp://ftp.ncbi.nih.gov/pub/HomoloGene/build$(HOMOLOGENE_VER)/
URL_IFILE=homologene.data
IFILE=$(FILE_CLASS).$(HOMOLOGENE_VER).data
DNLD=$(shell ../../web1/path_helper.py downloads)
FULL_IFILE=$(DNLD)$(IFILE)

BIOMART_URL=http://www.ensembl.org/biomart/martservice

# NOTE that if you're expanding these
# you'll also need to update the python code in this dir
# and likely the Salmon ETL (if you want to support RNA-seq)

# Mouse
M_NAME=mouse
M_NUMBER=10090
M_QUERY=$(M_NAME)_query.xml
M_E_C=$(M_NAME)_converter.tsv
# Rat
R_NAME=rat
R_NUMBER=10116
R_QUERY=$(R_NAME)_query.xml
R_E_C=$(R_NAME)_converter.tsv
# Dog
D_NAME=dog
D_NUMBER=9615
D_QUERY=$(D_NAME)_query.xml
D_E_C=$(D_NAME)_converter.tsv
# Zebrafish
Z_NAME=zebrafish
Z_NUMBER=7955
Z_QUERY=$(Z_NAME)_query.xml
Z_E_C=$(Z_NAME)_converter.tsv

XML_QUERIES=\
	$(M_QUERY) \
	$(R_QUERY) \
	$(D_QUERY) \
	$(Z_QUERY) \
	# end of list

CONVERTERS=\
	$(M_E_C) \
	$(R_E_C) \
	$(D_E_C) \
	$(Z_E_C) \
	# end of list


INPUT=\
	$(FULL_IFILE) \
	$(CONVERTERS) \
	# end of list


OFILE_MOUSE=$(FILE_CLASS).$(M_NAME).$(OUT_VER).entrez.tsv
OFILE_RAT=$(FILE_CLASS).$(R_NAME).$(OUT_VER).entrez.tsv
OFILE_DOG=$(FILE_CLASS).$(D_NAME).$(OUT_VER).entrez.tsv
OFILE_ZEBRAFISH=$(FILE_CLASS).$(Z_NAME).$(OUT_VER).entrez.tsv

ENSEMBLTRS_MOUSE=$(FILE_CLASS).$(M_NAME).$(OUT_VER).ensembltrs.tsv
ENSEMBLTRS_RAT=$(FILE_CLASS).$(R_NAME).$(OUT_VER).ensembltrs.tsv
ENSEMBLTRS_DOG=$(FILE_CLASS).$(D_NAME).$(OUT_VER).ensembltrs.tsv
ENSEMBLTRS_ZEBRAFISH=$(FILE_CLASS).$(Z_NAME).$(OUT_VER).ensembltrs.tsv

OFILES=\
	$(OFILE_MOUSE) \
	$(OFILE_RAT) \
	$(OFILE_DOG) \
	$(OFILE_ZEBRAFISH) \

ENSEMBL_TRS=\
	$(ENSEMBLTRS_MOUSE) \
	$(ENSEMBLTRS_RAT) \
	$(ENSEMBLTRS_DOG) \
	$(ENSEMBLTRS_ZEBRAFISH) \
	# end of list

OUTPUTS=\
	$(OFILES) \
	$(ENSEMBL_TRS) \
	# end of list


show_downloads:
	@echo $(INPUT)

show_latest_version:
	@echo 'Check the Homologene build at:'
	@echo ' https://ftp.ncbi.nih.gov/pub/HomoloGene/current/RELEASE_NUMBER'
	@echo 'Update HOMOLOGENE_VER in versions.py accordingly'
	@echo
	@echo "While it doesn't show up in the versions.py file,"
	@echo ' verify that the animal model genomes used in the Salmon ETL'
	@echo ' is the same as the latest Ensembl versions; see here:'
	@echo ' https://uswest.ensembl.org/info/about/species.html'
	@echo
	@echo 'Finish with the normal:'
	@echo ' make input, build, publish_s3'

input: $(INPUT)
build: $(OUTPUTS)


$(FULL_IFILE):
	curl $(URL)$(URL_IFILE) > temp
	mv temp $(FULL_IFILE)

$(XML_QUERIES):
	./$(XML_BUILD_PRGM)

%_converter.tsv:%_query.xml
	curl -d @$*_query.xml $(BIOMART_URL) > tmp.tmp
	mv tmp.tmp $*_converter.tsv

$(OFILES): $(FULL_IFILE)
	./$(PARSE_PRGM) $(FULL_IFILE) $(UNIPROT_VER) $(M_NUMBER) $(R_NUMBER) $(D_NUMBER) $(Z_NUMBER)
	mv $(M_NUMBER).tmp $(OFILE_MOUSE)
	mv $(R_NUMBER).tmp $(OFILE_RAT)
	mv $(D_NUMBER).tmp $(OFILE_DOG)
	mv $(Z_NUMBER).tmp $(OFILE_ZEBRAFISH)

$(FILE_CLASS).%.$(OUT_VER).ensembltrs.tsv:$(FILE_CLASS).%.$(OUT_VER).entrez.tsv %_converter.tsv
	./$(CNVRT_PRGM) $^ $(FILE_CLASS).$*.$(OUT_VER).ensembltrs.tsv

clean:
	-rm *.tsv
