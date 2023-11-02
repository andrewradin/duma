default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=grasp

DNLD=$(shell ../../web1/path_helper.py downloads)
WS=$(shell ../../web1/path_helper.py storage)

PARSE_PRGM=./parse_grasp.py
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)

URL=https://s3.amazonaws.com/NHLBI_Public/GRASP/
DFILE_UNZIPPED=GraspFullDataset$(GRASP_DNLD_VER)
DFILE=$(DFILE_UNZIPPED).zip
IFILE=$(OFILE_STEM).fullDataset
CHAIN_FILE=$(WS)hg19ToHg38.over.chain.gz

OFILE=$(OFILE_STEM).data.tsv.gz
STUDIES_FILE=$(OFILE_STEM).studies.tsv
FAILED_SNPS_FILE=$(OFILE_STEM).failed_snps.tsv
FAILED_STUDIES_FILE=$(OFILE_STEM).failed_studies.tsv

INPUT=$(DNLD)$(IFILE)

INPUTS=\
	$(INPUT) \
	$(CHAIN_FILE) \
	# end of list

OUTPUTS=\
	$(OFILE) \
	$(STUDIES_FILE) \
	$(FAILED_SNPS_FILE) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

$(CHAIN_FILE):
	../CrossMap/get_chain_files.py $(CHAIN)

$(INPUT):
	curl $(URL)$(DFILE) > $(DNLD)$(DFILE)
	unzip -o $(DNLD)$(DFILE)
	mv GRASP$(GRASP_DNLD_VER)fullDataset $(IFILE)
	dos2unix $(IFILE)
	mv $(IFILE) $(INPUT)


$(OUTPUTS): $(INPUTS)
	$(PARSE_PRGM) $(INPUT) $(STUDIES_FILE) $(FAILED_SNPS_FILE) $(FAILED_STUDIES_FILE) --chain_file $(CHAIN_FILE) > temp
	@#add some formatting to the FAILED_STUDIES_FILE so we don't accidentally catch snp positions
	awk '{print "|" $$0 "\t"}' $(FAILED_STUDIES_FILE) > fsf.tmp
	grep -F -v -f fsf.tmp temp > stripmaxpheno.tmp
	gzip -c stripmaxpheno.tmp > $(OFILE)
	rm -f temp *.tmp

clean:
	rm -rf *.tsv *.tsv.gz temp *txt *log *.tmp
