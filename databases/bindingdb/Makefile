default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=bindingdb

include ../version_tools/setup.mk
include versions.mk

TMP_OUT=tmp_out.tmp

C50_PRGM=../matching/clean_simple.py
PRS_PRGM=./process_bindingDB_data.py

WS_DNLD=$(shell ../../web1/path_helper.py downloads)

RAW_ZIPPED=BindingDB_All_$(BINDINGDB_VER).tsv.zip
RAW_FILE=BindingDB_All.tsv
INPUT=$(WS_DNLD)/BindingDB_All_$(BINDINGDB_VER).tsv.gz
ARCHIVED_INPUTS=$(INPUT)

# BindingDB should be extracted last of all the collection datasets, so
# the others are available as inputs to the local match_tool run (which
# is used for condensing).
input: $(INPUT)

show_downloads:
	@echo $(INPUT)

show_latest_version:
	@echo
	@echo 'To find the latest version number to add to the versions.py file,'
	@echo 'go to https://www.bindingdb.org and click the Download tab. Use'
	@echo 'the YYYYm<number> string that appears in all the'
	@echo '"Ligand-Target-Affinity Datasets" filenames.'
	@echo
	@echo 'Finally update to the next matching version (the current +1)'
	@echo 'This will be used and executed only in the matching directory.'
	@echo 'As a result run only the following:'
	@echo 'make input'
	@echo 'make pre_cluster_build'


$(INPUT):
	# if the following fails for re-generating an older version,
	# the file can be retrieved from S3 via:
	# $(S3_TOOL) --downloads-archive $(FILE_CLASS) $(notdir $(INPUT))
	wget 'https://www.bindingdb.org/bind/downloads/$(RAW_ZIPPED)'
	unzip $(RAW_ZIPPED)
	rm $(RAW_ZIPPED)
	gzip -c $(RAW_FILE) >$(TMP_OUT)
	rm $(RAW_FILE)
	mv $(TMP_OUT) $(INPUT)

ATTR_OUTPUT=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
CONDENSED_ATTR_OUTPUT=$(FILE_CLASS).full_condensed.$(OUT_VER).attributes.tsv

ASSAY_C50=$(FILE_CLASS).c50.$(OUT_VER).unmapped_affinity.tsv
ASSAY_KI=$(FILE_CLASS).ki.$(OUT_VER).unmapped_affinity.tsv
SRCS_EXTR=$(FILE_CLASS).full.$(OUT_VER).srcs.tsv
ASSAY_EXTR=$(ASSAY_C50) $(ASSAY_KI)
CLEANED_ASSAYS=$(subst unmapped_affinity,affinity,$(ASSAY_EXTR))
EVIDENCE_OUTPUTS=$(subst unmapped_affinity,evidence,$(ASSAY_EXTR))
MERGED_DPI=$(FILE_CLASS).merged.$(OUT_VER).evidence.tsv

ifdef MATCHING_VER
CLUSTER_FILE=matching.full.$(MATCHING_VER).clusters.tsv
CLUSTER_PATH=$(shell ../matching/find_etl_file.py $(CLUSTER_FILE))
else
# if running an older or erroneous configuration that doesn't define
# MATCHING_VER, this will result in an easier-to-interpret error
CLUSTER_PATH=UNDEFINED_CLUSTER_PATH
endif

PARSE_OUTPUTS=\
	$(ATTR_OUTPUT) \
	$(ASSAY_EXTR) \
	$(SRCS_EXTR) \
	# end of list

OUTPUTS=\
	$(ATTR_OUTPUT) \
	$(CONDENSED_ATTR_OUTPUT) \
	$(EVIDENCE_OUTPUTS) \
	$(CLEANED_ASSAYS) \
	# end of list

pre_cluster_build:$(ATTR_OUTPUT)
build: $(OUTPUTS)

SELECTED=tmp.$(BINDINGDB_VER).selected.tsv
$(SELECTED): $(INPUT)
	echo -e "smiles_code\tinchi\tinchi_key\tbindingdb_id\tcanonical\tKi\tIC50\tEC50\tcuration_id\tarticle_doi\tpubmed_id\tpubchem_aid\tpatent_id\tpubchem_cid\tChEMBL_ID\tdrugbank_id\tkegg\tPrimary_Uniprot_ID\tAlternative_Uniprot_IDs" > $(SELECTED)
	zgrep "_HUMAN" $(INPUT) | cut -f2-6,9,10,12,17-21,29,32,33,35,42,43 | sort -u >> $(SELECTED)

PROTMAP_VER=HUMAN_9606.$(UNIPROT_VER)
$(PARSE_OUTPUTS): $(SELECTED)
	$(PRS_PRGM) -i $(SELECTED) -u $(PROTMAP_VER)
	../matching/make_std_smiles.py -r worker-test -i ds.tmp -o $(ATTR_OUTPUT)
	mv c50.tmp $(ASSAY_C50)
	mv ki.tmp $(ASSAY_KI)
	mv srcs.tmp $(SRCS_EXTR)

$(MERGED_DPI): $(EVIDENCE_OUTPUTS)
	../matching/dpi_merge.py $(MERGED_DPI) $(EVIDENCE_OUTPUTS)

$(CONDENSED_ATTR_OUTPUT): $(ATTR_OUTPUT) \
		$(SRCS_EXTR) $(CLEANED_ASSAYS) $(CLUSTER_PATH) $(MERGED_DPI)
	./condense_bindingdb.py $(SRCS_EXTR) $(CLEANED_ASSAYS) $(CLUSTER_PATH) \
			$(MERGED_DPI) $@

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

clean:
	-rm *.tsv

