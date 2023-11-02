default:
	@echo "USAGE: make input|all|clean"

SEARCH_TYPE=CC

ODIR=$(shell ../../web1/path_helper.py storage)

TRAINING_SAMPLES_FILE=allExistingGEDatasets.tsv
INPUTS=\
	$(TRAINING_SAMPLES_FILE) \
	# end of list

PREFIX=AE_$(SEARCH_TYPE)_search
LOG=$(PREFIX)_build.log
MODEL=$(PREFIX).model
FINAL_MODEL=$(ODIR)$(MODEL)
PICKLE=$(PREFIX).pickle
DATA_PKL=$(PREFIX)_data.pickle
FINAL_PICKLE=$(ODIR)$(PICKLE)
ARFF=$(PREFIX).arff
OUTPUTS=\
	$(MODEL) \
	$(PICKLE) \
	# end of list

PUBS=\
	$(FINAL_MODEL) \
	$(FINAL_PICKLE) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)
publish: $(PUBS)
publish_s3: publish
all:
	$(MAKE) publish_s3
	$(MAKE) publish_s3 SEARCH_TYPE=TR

$(TRAINING_SAMPLES_FILE):
	./extract.py > temp
	mv temp $(TRAINING_SAMPLES_FILE)

$(OUTPUTS): $(INPUTS)
	./build.py $(TRAINING_SAMPLES_FILE) $(SEARCH_TYPE) -o temp --opkl $(DATA_PKL) > $(LOG)
	mv temp $(ARFF)
	./build_model.py $(ARFF) $(PREFIX)

$(PUBS):$(OUTPUTS)
	mv $(MODEL) $(FINAL_MODEL)
	mv $(PICKLE) $(FINAL_PICKLE)

publish_s3:
	s3cmd put $(FINAL_MODEL) s3://duma-datasets/$(MODEL)
	s3cmd put $(FINAL_PICKLE) s3://duma-datasets/$(PICKLE)

clean:
	rm -rf *.tsv *.tsv.gz temp *pkl *arff
