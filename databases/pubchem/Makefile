default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=pubchem

include ../version_tools/setup.mk
include versions.mk

WS_DNLD=$(shell ../../web1/path_helper.py downloads)


CAS_MAP=cid_to_cas.$(OUT_VER).json
ATTR_FILE_FULL=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
ATTR_FILE_FILTERED=$(FILE_CLASS).filtered.$(OUT_VER).attributes.tsv

ifdef MATCHING_VER
CLUSTER_FILE=matching.full.$(MATCHING_VER).clusters.tsv
CLUSTER_PATH=$(shell ../matching/find_etl_file.py $(CLUSTER_FILE))
else
# if running an older or erroneous configuration that doesn't define
# MATCHING_VER, this will result in an easier-to-interpret error
CLUSTER_PATH=UNDEFINED_CLUSTER_PATH
endif

OUTPUTS=\
	$(ATTR_FILE_FULL) \
	$(ATTR_FILE_FILTERED) \
	# end of list

TMP_IN=in.tmp
TMP_OUT=out.tmp

show_downloads:
	@echo $(INPUT)

show_latest_version:
	@echo
	@echo 'All that is need is to update to the next matching version (the current +1)'
	@echo 'This will be used and executed only in the matching directory.'
	@echo 'As a result run only the following:'
	@echo 'make pre_cluster_build'
	@echo
	@echo 'NOTE: there is no make input, the files are downloaded and processed'
	@echo '      "on the fly" during the pre_cluster_build.'

pre_cluster_build: $(ATTR_FILE_FULL)
build: $(OUTPUTS)

$(CAS_MAP):
	./fetch_pubchem_cas.py -o $(CAS_MAP)

$(ATTR_FILE_FULL): $(CAS_MAP)
	./make_pubchem.py -o nostd.$(ATTR_FILE_FULL) --cas-map $(CAS_MAP)
	../matching/make_std_smiles.py -r worker-test -i nostd.$(ATTR_FILE_FULL) -o $(TMP_OUT)
	mv $(TMP_OUT) $(ATTR_FILE_FULL)

$(ATTR_FILE_FILTERED): $(CLUSTER_PATH) $(ATTR_FILE_FULL)
	./filter_pubchem.py -i $(ATTR_FILE_FULL) -c $(CLUSTER_PATH) -o $(ATTR_FILE_FILTERED)

clean:
	-rm *.tsv
