

INPUTS=\
	   tier1.tsv\
	   tier2.tsv \
	   # end of list

OUTPUT=global_protsets.tsv
S3_STORAGE=s3://duma-datasets/

build:	$(OUTPUT)

$(OUTPUT): $(INPUTS)
	./build.py -o $(OUTPUT)

clean:
	-rm $(OUTPUT)

publish_s3:	$(OUTPUT)
	s3cmd put $(OUTPUT) $(S3_STORAGE)$(OUTPUT)
