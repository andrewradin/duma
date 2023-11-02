default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=chembl

include ../version_tools/setup.mk
include versions.mk

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

show_latest_version:
	@echo
	@echo '=============================================='
	@echo 'Visit https://www.ebi.ac.uk/chembl/ to find the current ChEMBL version'
	@echo 'Update CHEMBL_VER in versions.py accordingly.'
	@echo
	@echo 'Also update Uniprot to our latest version'
	@echo
	@echo 'Finally update to the next matching version (the current +1)'
	@echo 'This will be used and executed only in the matching directory.'
	@echo 'As a result run only the following:'
	@echo 'make input'
	@echo 'make pre_cluster_build'


######
# Theory of operation
# - 'input' target downloads data, creates a mysql database and creates a
#   peewee schema file for accessing the database
# - 'build' target extracts c50 and ki data, normalizes it, uses the list
#   of chembl ids present in the normalized data to create raw attribute
#   files, and then normalizes those. An admeAssays and pcAssays files are
#   created as by-products of attribute extraction.
# - output file names have the general form:
#   chembl.flavor.vX.role.fmt
#   - for DPI data, flavor is either c50 or ki, and role is either
#     unmapped_affinity or affinity
#   - for attribute data, flavor is either full or adme, and role is either
#     no_std_smiles or attributes
######

######
# download a new version
# - to find the latest version for the versions.py entry, go to $(URL_BASE)
#   in a browser, and find the last entry in the releases subdirectory
######
VERNAME=chembl_$(CHEMBL_VER)
URL_BASE=ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb
#URL=$(URL_BASE)/latest
# using 'latest' as above works for the latest, but the one below works
# for any version (and you need the version number to download in any case)
URL=$(URL_BASE)/releases/$(VERNAME)
DNLD_FILE=$(VERNAME)_mysql.tar.gz
UNTAR_DIR=$(VERNAME)

$(WS_DNLD)$(DNLD_FILE):
	curl $(URL)/$(DNLD_FILE) >$@

$(UNTAR_DIR): $(WS_DNLD)$(DNLD_FILE)
	tar xzf $(WS_DNLD)$(DNLD_FILE)

MYSQL=mysql -u root
DB_NAME=$(VERNAME)
SELECT=select schema_name from information_schema.schemata
WHERE=where schema_name='$(DB_NAME)'
PROBE=$(MYSQL) -s -N -e "$(SELECT) $(WHERE)"
DB_OPTS=DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci
DUMPFILE=$(UNTAR_DIR)/$(VERNAME)_mysql/$(VERNAME)_mysql.dmp
SCHEMA=$(VERNAME)_schema.py


# Since $(SCHEMA) doesn't depend on anything, once it exists we can
# remove the intermediate untarred download, but by invoking make on
# $(UNTAR_DIR) before trying to create the schema file, we'll rebuild
# the untar directory if it's been removed or hasn't been created.
$(SCHEMA):
	if [ -z `$(PROBE)` ] ; then \
		$(MAKE) $(UNTAR_DIR); \
		$(MYSQL) -e "create database $(DB_NAME) $(DB_OPTS)" ;\
		$(MYSQL) $(DB_NAME) < $(DUMPFILE);\
	fi
	bash peewee.sh $(DB_NAME) > pwiz_output
	mv pwiz_output $@

input: $(SCHEMA)

######
# DPI extract
######
# order is important: must match extract_potency_data_chembl.py parm order
ASSAY_FLAVORS=c50 ki

ASSAY_OUTPUT_PATTERN=$(FILE_CLASS).%.$(OUT_VER).affinity.tsv
FULL_ASSAY_RAW=$(FILE_CLASS).default.$(OUT_VER).raw_dpi_assays.tsv
ASSAY_OUTPUTS=$(patsubst %,$(ASSAY_OUTPUT_PATTERN),$(ASSAY_FLAVORS))
ASSAY_RAW=$(subst affinity,unmapped_affinity,$(ASSAY_OUTPUTS))
ASSAY_EV=$(subst affinity,evidence,$(ASSAY_OUTPUTS))
DPI_FOR_CONDENSE=$(FILE_CLASS).default.$(OUT_VER).dpi.tsv

PROTMAP_VER=HUMAN_9606.$(UNIPROT_VER)

$(ASSAY_RAW) $(FULL_ASSAY_RAW): $(SCHEMA)
	./extract_potency_data_chembl.py $(ASSAY_RAW) $(FULL_ASSAY_RAW) $(VERNAME)

%.affinity.tsv: %.unmapped_affinity.tsv
	../matching/map_uniprot_names.py $< $*.mapped.tmp $(PROTMAP_VER)
	../matching/clean_simple.py $*.mapped.tmp $*.cleaned_output.tmp
	rm $*.mapped.tmp
	mv $*.cleaned_output.tmp $@

%.evidence.tsv: %.affinity.tsv
	../matching/aff_to_ev.py $< ev.tmp
	head -1 ev.tmp > evsorted.tmp
	tail -n +2 ev.tmp | sort >> evsorted.tmp
	rm ev.tmp
	mv evsorted.tmp $@

$(DPI_FOR_CONDENSE): $(ASSAY_EV)
	../matching/dpi_merge.py $@ $(ASSAY_EV)

######
# Attribute extract
######
# all collection subsets
ATTR_SUBSETS=full adme adme_condensed
ATTR_OUTPUT_PATTERN=$(FILE_CLASS).%.$(OUT_VER).attributes.tsv
ATTR_OUTPUTS=$(patsubst %,$(ATTR_OUTPUT_PATTERN),$(ATTR_SUBSETS))
# collection subsets produced by parseChEMBL2.py
FULL_ATTR_RAW=$(FILE_CLASS).full.$(OUT_VER).no_std_smiles.tsv
FULL_ATTR=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
# collection subsets produced by extract_adme_comm_subsets.py
ADME_ATTR=$(FILE_CLASS).adme.$(OUT_VER).attributes.tsv

IND_FILE=$(FILE_CLASS).full.$(OUT_VER).indications.tsv

ADME_ASSAYS=$(FILE_CLASS).adme.$(OUT_VER).adme_assays.tsv.gz
PC_ASSAYS=$(FILE_CLASS).full.$(OUT_VER).pc_assays.tsv.gz
TOX_ASSAYS=$(FILE_CLASS).full.$(OUT_VER).tox_assays.tsv.gz
ALL_TSV_ASSAYS=$(ADME_ASSAYS) $(PC_ASSAYS) $(TOX_ASSAYS)
ALL_SQL_ASSAYS=$(subst .tsv.gz,.sqlsv.gz,$(ALL_TSV_ASSAYS))

DPI_FILT_FILES=$(ASSAY_OUTPUTS)

ifdef MATCHING_VER
CLUSTER_FILE=matching.full.$(MATCHING_VER).clusters.tsv
CLUSTER_PATH=$(shell ../matching/find_etl_file.py $(CLUSTER_FILE))
else
# if running an older or erroneous configuration that doesn't define
# MATCHING_VER, this will result in an easier-to-interpret error
CLUSTER_PATH=UNDEFINED_CLUSTER_PATH
endif


$(FULL_ATTR_RAW) $(ALL_TSV_ASSAYS): $(SCHEMA) $(DPI_FILT_FILES) $(IND_FILE)
	cat $(DPI_FILT_FILES) | cut -f 1 | sort -u > temp.tsv
	cat $(IND_FILE) | cut -f1 | sort -u >> temp.tsv
	./parseChEMBL2.py --all --requireStructure temp.tsv $(VERNAME)
	rm temp.tsv
	mv ds.full.tmp $(FILE_CLASS).full.$(OUT_VER).no_std_smiles.tsv
	gzip -c admeAssays.tmp > $(ADME_ASSAYS)
	gzip -c pcAssays.tmp > $(PC_ASSAYS)
	gzip -c toxAssays.tmp > $(TOX_ASSAYS)
	rm -f admeAssays.tmp pcAssays.tmp toxAssays.tmp

%_assays.sqlsv.gz: %_assays.tsv.gz
	../../web1/scripts/tsv_convert.py \
		-i $< -o $(subst .gz,,$@) \
		--index "chembl_id assay_chembl_id" \
		str str str float str str str str
	gzip $(subst .gz,,$@)

%.attributes.tsv: %.no_std_smiles.tsv
	../matching/make_std_smiles.py -i $< -o $@ -r worker-test

$(ADME_ATTR): $(FULL_ATTR) $(CLUSTER_PATH) $(DPI_FOR_CONDENSE)
	./extract_adme_comm_subset.py \
		-o $@ \
		--full $(FULL_ATTR) \
		--clusters $(CLUSTER_PATH) \
		--adme $(ADME_ASSAYS) \
		--dpi $(DPI_FOR_CONDENSE)

%_condensed.$(OUT_VER).attributes.tsv: %.$(OUT_VER).attributes.tsv \
		$(DPI_FOR_CONDENSE)
	./condense_chembl.py $(VERNAME) $(DPI_FOR_CONDENSE) $@

######
# Indication extract
######

$(IND_FILE): $(SCHEMA)
	./extract_indications.py $(IND_FILE) $(VERNAME)

######
# Output roll-up
######
OUTPUTS=\
	$(ASSAY_OUTPUTS) \
	$(ASSAY_EV) \
	$(ATTR_OUTPUTS) \
	$(ALL_TSV_ASSAYS) \
	$(IND_FILE) \
    $(FULL_ASSAY_RAW) \
	$(ALL_SQL_ASSAYS) \
    # end of list


pre_cluster_build: $(FULL_ATTR)
build: $(OUTPUTS)

clean:
	rm -rf *.tsv *pdf *tmp

