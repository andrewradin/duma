default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=reactome
include ../version_tools/setup.mk
include versions.mk


SOURCE=reactome

GRAPH_URL=https://reactome.org/download/$(REACTOME_VER)/reactome.graphdb.tgz
DIAGRAMS_URL=https://reactome.org/download/$(REACTOME_VER)/diagrams.svg.tgz
GRAPH_FILE_NAME=reactome.$(OUT_VER).graphdb.tgz
DIAGRAMS_FILE_NAME=reactome.$(OUT_VER).diagrams.tgz

GENESETS=reactome.$(OUT_VER).genesets.tsv
HIER=reactome.$(OUT_VER).hierarchy.tsv

WS=$(shell ../../web1/path_helper.py storage)
UNI_FILE=uniprot.HUMAN_9606.v$(UNIPROT_VER).Uniprot_data.tsv
UNI_BUCKET=uniprot
UNIPROT_CONVERTER=$(WS)/$(UNI_BUCKET)/$(UNI_FILE)

INPUTS=\
	$(DIAGRAMS_FILE_NAME) \
	$(GRAPH_FILE_NAME) \
    # end of list

OUTPUTS=\
	$(DIAGRAMS_FILE_NAME) \
	$(GENESETS) \
	$(HIER) \
    # end of list

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'Determine if a new Reactome version has been released by'
	@echo ' visting https://reactome.org/'
	@echo 'Compare the listed version to REACTOME_VER in versions.py'
	@echo 'If there is an update, then run:'
	@echo make input
	@echo make build
	@echo make publish_s3
	@echo

$(GRAPH_FILE_NAME) $(DIAGRAMS_FILE_NAME):
	curl -f -L $(GRAPH_URL) > $(GRAPH_FILE_NAME)
	curl -f -L $(DIAGRAMS_URL) > $(DIAGRAMS_FILE_NAME)

$(GENESETS) $(HIER): $(GRAPH_FILE_NAME)
	./run_graphdb.sh $(GRAPH_FILE_NAME) $(WS)/downloads/rct-neo4j/$(REACTOME_VER)/
	./parse_reactome.py \
		--uniprot $(UNIPROT_CONVERTER) \
		--all $(GENESETS) \
		--hier $(HIER)
	sudo docker rm -f neo4j-reactome

clean:
	-rm *.tgz
