default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=gene_ontology

include ../version_tools/setup.mk
include versions.mk

WS=$(shell ../../web1/path_helper.py storage)
WS_DNLD=$(shell ../../web1/path_helper.py downloads)

# GO comes in two parts, the ontology itself which describes pathways, molecular functions
# and cell locations as a graph, and then the annotations (GOA) which associate things (e.g. genes)
# to those ontology terms.
# GO also maintains a Reactome mapping file, which might be useful.
GO_URL=http://purl.obolibrary.org/obo/go.obo
GOA_URL=http://geneontology.org/gene-associations/goa_human.gaf.gz
GO2REACTOME_URL=http://current.geneontology.org/ontology/external2go/reactome2go

GO_FILE=go.$(GO_VER).obo
GOA_FILE=goa.$(GO_VER).gaf.gz

UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
ENSEMBL_UNIPROT_CONVERTER=$(WS)$(UNI_BUCKET)/$(UNI_FILE)

GO_GENESETS=$(FILE_CLASS).$(OUT_VER).genesets.tsv
GO_HIERARCHY=$(FILE_CLASS).$(OUT_VER).hierarchy.tsv

INPUT=\
	$(GO_FILE) \
	$(GOA_FILE) \
	# end of list

OUTPUTS=\
	$(GO_GENESETS) \
	$(GO_HIERARCHY) \
	# end of list

show_downloads:
	@echo $(INPUT)

input: $(GO_FILE) $(GOA_FILE)

build: $(OUTPUTS)


$(GO_GENESETS) $(GO_HIERARCHY): $(GO_FILE) $(GOA_FILE)
	./parse.py \
		-g $(GO_FILE) -a $(GOA_FILE) \
		--out-genesets $(GO_GENESETS) --out-hier $(GO_HIERARCHY) \
		-u $(ENSEMBL_UNIPROT_CONVERTER)

$(GO_FILE):
	curl -L $(GO_URL) > tmp.tmp
	mv tmp.tmp $(GO_FILE)

$(GOA_FILE):
	curl -L $(GOA_URL) > tmp.tmp
	mv tmp.tmp $(GOA_FILE)



show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Visit http://current.geneontology.org/products/pages/downloads.html and update GO_VER to the release date (e.g. 2021-02-01)'
	@echo 'make input'
	@echo 'make build'
	@echo 'make publish_s3'

clean:
	-rm *.tsv *.tmp *.owl *.gz

