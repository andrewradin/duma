default:
	@echo "USAGE: make input|build|clean"

PRGM=./create_dpi_bg_scores.py

ODIR=$(shell ../../web1/path_helper.py storage)
IDIR=$(shell ../../web1/path_helper.py storage)dpi

INPUTS=$(shell echo $(IDIR)/dpi.*.tsv)
DPIS=$(patsubst $(IDIR)/dpi.%.tsv,%,$(INPUTS))

OUTPUTS=$(patsubst %,dpi.%.bg.tsv,$(DPIS))

PUBS=$(patsubst %,$(ODIR)/%,$(OUTPUTS))

build: $(OUTPUTS)
publish: $(PUBS)

dpi.%.bg.tsv: $(IDIR)/dpi.%.tsv
	$(PRGM) --dpi_name $* > temp
	mv temp $@

$(ODIR)/%: %
	cp $* $@

publish_s3: $(PUBS)
	@for FILE in $(OUTPUTS);\
	do \
		s3cmd put $$FILE s3://duma-datasets/$$FILE ; \
	done

clean:
	-rm -rf *.tsv *.txt
