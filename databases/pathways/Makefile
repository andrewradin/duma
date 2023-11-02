default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=pathways
include ../version_tools/setup.mk
include versions.mk

ALL_GENESETS=pathways.combined.all.$(OUT_VER).genesets.tsv
ALL_HIERARCHY=pathways.combined.all.$(OUT_VER).hierarchy.tsv
ALL_GENE2PATH=pathways.combined.all.$(OUT_VER).gene_to_pathway.sqlsv

WS=$(shell ../../web1/path_helper.py storage)
REACTOME_GS=$(WS)reactome/reactome.v$(REACTOME_VER).genesets.tsv
REACTOME_HIER=$(WS)reactome/reactome.v$(REACTOME_VER).hierarchy.tsv

GENE_ONTOLOGY_GS=$(WS)/gene_ontology/gene_ontology.v$(GENE_ONTOLOGY_VER).genesets.tsv
GENE_ONTOLOGY_HIER=$(WS)/gene_ontology/gene_ontology.v$(GENE_ONTOLOGY_VER).hierarchy.tsv

OUTPUTS=\
	$(ALL_GENESETS) \
	$(ALL_HIERARCHY) \
	$(ALL_GENE2PATH) \
    # end of list

show_latest_version:
	@echo
	@echo '============================================='
	@echo 'As needed, update versions.py with the latest version of:'
	@echo ' our reactome version number'
	@echo ' our gene_ontology version number'
	@echo
	@echo 'After that, run:'
	@echo 'make build'
	@echo 'make publish_s3'
	@echo
	@echo 'NOTE that this dir has no make input.'

build: $(OUTPUTS)

$(ALL_GENESETS) $(ALL_HIERARCHY) $(ALL_GENE2PATH): $(REACTOME_GS) $(REACTOME_HIER) $(GENE_ONTOLOGY_GS) $(GENE_ONTOLOGY_HIER)
	./merge_genesets.py \
		--genesets $(REACTOME_GS) $(GENE_ONTOLOGY_GS) \
		--hierarchies $(REACTOME_HIER) $(GENE_ONTOLOGY_HIER) \
		--output-geneset $(ALL_GENESETS) \
		--output-hierarchy $(ALL_HIERARCHY) \
		--output-gene-to-pathway $(ALL_GENE2PATH)


clean:
	-rm *.tsv
