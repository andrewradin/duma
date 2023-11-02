default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

# This directory produces two kinds of things:
# - several conversion tsv files, which map between uniprot names and various
#   other protein vocabularies
# - a gzipped json file which holds gene name information for a protein
#
# Although uniprot serves old releases off their website, they're not in
# the same format as the current release. This ETL relies on the current
# release format for the tsv outputs, and uses a database query for the
# json output, so it isn't really worth trying to re-create old inputs.
# Instead, inputs are archived on S3 in the expected form.

FILE_CLASS=uniprot

SPECIESNAME=HUMAN
SPECIESNUMBER=9606

CUR_RELEASE_URL=ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/
METALINK_URL=$(CUR_RELEASE_URL)RELEASE.metalink
BY_ORG_URL=$(CUR_RELEASE_URL)knowledgebase/idmapping/by_organism/
WEBFILE=$(SPECIESNAME)_$(SPECIESNUMBER)_idmapping.dat.gz

NAMES_QUERY_URL="https://rest.uniprot.org/uniprotkb/stream?query=%28%28organism_id%3A9606%29%29&compressed=true&fields=accession%2Cid%2Cprotein_name%2Cgene_names&format=xml"
CANON_QUERY_URL="https://rest.uniprot.org/uniprotkb/stream?query=%28%28organism_id%3A9606%29%29&compressed=true&fields=accession%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Cgene_primary%2Cgene_synonym%2Creviewed%2Cxref_ensembl%2Cxref_hgnc&format=tsv"

FILE_STEM=$(FILE_CLASS).$(SPECIESNAME)_$(SPECIESNUMBER)

DNLD=$(shell ../../web1/path_helper.py downloads)

IDMAP_INPUT=$(DNLD)$(FILE_STEM).$(UNIPROT_VER).idmapping.dat.gz
NAMES_INPUT=$(DNLD)$(FILE_STEM).$(UNIPROT_VER).names.xml.gz
CANON_INPUT=$(DNLD)$(FILE_STEM).$(UNIPROT_VER).canonical.tsv.gz
ARCHIVED_INPUTS=$(IDMAP_INPUT) $(NAMES_INPUT)

TMP=tempfile.tmp
TMP2=tempfile2.tmp

OFILE_STEM=$(FILE_STEM).$(OUT_VER)
OFILE_MAIN=$(OFILE_STEM).Uniprot_data.tsv
OFILE1=$(OFILE_STEM).Protein_Entrez.tsv
OFILE2=$(OFILE_STEM).Protein_EnsemblTRS.tsv
OFILE3=$(OFILE_STEM).Protein_Ensembl.tsv
OFILE_NAMES=$(OFILE_STEM).Protein_Names.json.gz

MAIN_OUTPUTS=\
	$(OFILE_MAIN) \
	$(OFILE1) \
	$(OFILE2) \
	$(OFILE3) \
	# end of list

OUTPUTS=$(MAIN_OUTPUTS) $(OFILE_NAMES)

show_downloads:
	@echo $(IDMAP_INPUT)
	@echo $(NAMES_INPUT)
	@echo $(CANON_INPUT)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Determine if a new Uniprot version has been released by'
	@echo ' visting https://www.uniprot.org/'
	@echo 'Compare the listed version to UNIPROT_VER in versions.py'
	@echo 'If there is an update, then run:'
	@echo make input
	@echo make build
	@echo make publish_s3

build: $(OUTPUTS) $(OFILE_NAMES)

# Since the version is not explicitly associated with either input file,
# we need to retrieve both inputs and do the version check in a single step.
# The build steps depend on the individual files, which must be retrieved
# manually (either via make input or from S3). The following provides
# instructions if the appropriate version of the input files are not present.
#
# Since the input retrieval is actually scripted in the 'input' make target,
# we don't use get_s3_input_or_prompt here.
INPUT_ERROR_MSG="\n"
INPUT_ERROR_MSG+="\n"
INPUT_ERROR_MSG+="get uniprot version $(UNIPROT_VER)\n"
INPUT_ERROR_MSG+="(from S3 or with 'make input')\n"
INPUT_ERROR_MSG+="\n"
INPUT_ERROR_MSG+="\n"
$(IDMAP_INPUT) $(NAMES_INPUT) $(CANON_INPUT):
	@echo $(INPUT_ERROR_MSG)  && false

input:
	curl $(METALINK_URL) > $(TMP)
	./check_version.py $(UNIPROT_VER) $(TMP)
	curl $(BY_ORG_URL)$(WEBFILE) > $(TMP)
	mv $(TMP) $(IDMAP_INPUT)
	curl -H "Accept: application/xml" $(NAMES_QUERY_URL) > $(TMP)
	mv $(TMP) $(NAMES_INPUT)
	curl -H "Accept: text/plain; format=tsv" $(CANON_QUERY_URL) > $(TMP)
	mv $(TMP) $(CANON_INPUT)


$(MAIN_OUTPUTS): $(IDMAP_INPUT) $(CANON_INPUT) $(NAMES_INPUT)
	./uniprot_parser2.py -i $(IDMAP_INPUT) -c $(CANON_INPUT) -x $(NAMES_INPUT) -o $(OFILE_MAIN) -d debug.tsv
	cat $(OFILE_MAIN) | grep GeneID | cut -f 1,3 > $(TMP)
	mv $(TMP) $(OFILE1)
	grep Ensembl_TRS $(OFILE_MAIN) | perl -lane 'print "$$F[2]\t$$F[0]";' > $(TMP)
	mv $(TMP) $(OFILE2)
	cat $(OFILE_MAIN) | grep Ensembl | cut -f 1,3 > $(TMP)
	mv $(TMP) $(OFILE3)

$(OFILE_NAMES): $(NAMES_INPUT)
	zcat $(NAMES_INPUT) | ./uniprot_name_parse.py | gzip > $(TMP)
	mv $(TMP) $@

clean:
	-rm *.tsv
	-rm *.json.gz
	-rm $(TMP) $(TMP2)
