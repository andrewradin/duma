default:
	@echo "USAGE: make input|build|clean"

URL='https://raw.githubusercontent.com/OHDSI/Vocabulary-v4.5/661804cf3c17add61b02e2e83e477f48acb011d5/21-ATC/'

IFILE=atc_code.txt
	
ODIR=$(shell ../../web1/path_helper.py storage)
DNLD=$(shell ../../web1/path_helper.py downloads)

OFILE_MAIN=atc_code.tsv
OPATH_MAIN=$(ODIR)/$(OFILE_MAIN)

INPUT=$(DNLD)$(IFILE)

OUTPUTS=$(OFILE_MAIN)

PUBS=$(OPATH_MAIN)



show_downloads:
	@echo $(INPUT)

input: $(INPUT)
build: $(OUTPUTS)
publish: $(PUBS)

$(INPUT):
	curl $(URL)$(IFILE) > $(INPUT)

$(OUTPUTS): $(INPUT)
	./process_atc.py $(INPUT) $(OFILE_MAIN)
	cp $(OFILE_MAIN) $(OPATH_MAIN)

publish_s3: $(PUBS)
	s3cmd put $(OPATH_MAIN) s3://duma-datasets/$(OFILE_MAIN)

clean:
	-rm *.txt

