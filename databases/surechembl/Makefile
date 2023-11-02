default:
	@echo "USAGE: make input|build|clean"

PRS_PRGM=parse_surechembl.py
COLLECTION=surechembl
URL=ftp://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/data/map/
S3_DRUGSETS=s3://2xar-duma-drugsets/
S3_STORAGE=s3://duma-datasets/
ODIR=$(shell ../../web1/path_helper.py storage)
DRUGSETS_DIR=$(ODIR)drugsets/
DNLD=$(shell ../../web1/path_helper.py downloads)
BIGTMP=$(shell ../../web1/path_helper.py bigtmp)

OCF_PREF=create.
FF_SUF=.full.tsv.gz
OPF_PREF=patent.
OFILE_CREATE=$(OCF_PREF)$(COLLECTION)$(FF_SUF)
OFILE_PATENT=$(OPF_PREF)$(COLLECTION)$(FF_SUF)
OPATH_CREATE=$(ODIR)/$(OFILE_CREATE)
OPATH_PATENT=$(ODIR)/$(OFILE_PATENT)

FILES=$(shell curl $(URL) | tr ' ' '\n' | grep 'SureChEMBL_map' - | tr '\n' ' ')
#FILES= \
#	SureChEMBL_map_20150401.txt.gz \
#	SureChEMBL_map_20150701.txt.gz \
#	SureChEMBL_map_20180401.txt.gz \

DFILES=$(foreach FILE, $(FILES), $(DNLD)$(FILE))
IOFS_CREATE=$(foreach FILE, $(FILES), $(OCF_PREF)$(FILE))
IOPS_CREATE=$(foreach FILE, $(FILES), $(OPF_PREF)$(FILE))

STARTING_COLLECTIONS= \
	drugbank \
	chembl \
	# end of list

SC_LIST=$(shell echo $(STARTING_COLLECTIONS) | tr ' ' ',')

# this has the known issue that not all collections
# will come before surechembl in the unichem hierarchy like ChEMBL and DrugBank do.
# If/when that becomes an issue, we'll fix it
TFILES=$(foreach SC, $(STARTING_COLLECTIONS), $(SC)_to_surechembl.tsv.gz)
TF_LIST=$(shell echo $(TFILES) | tr ' ' ',')

CFILE_PREF=create.
CFILE_SUF=.full.tsv
CFILE_NAMES=$(foreach SC, $(STARTING_COLLECTIONS), $(CFILE_PREF)$(SC)$(CFILE_SUF))
CFILES=$(foreach CFN, $(CFILE_NAMES), $(DRUGSETS_DIR)$(CFN))

RFILE_SUF=_oi.tsv
RFILES=$(foreach SC, $(STARTING_COLLECTIONS), $(SC)$(RFILE_SUF))
RF_LIST=$(shell echo $(RFILES) | tr ' ' ',')

show_downloads:
	@echo $(FILES)

show_dest:
	@echo $(DFILES)

INPUTS=\
	$(DFILES) \
	$(TFILES) \
	$(RFILES) \
	# end of list

OUTPUTS=\
	$(OFILE_CREATE) \
	$(OFILE_PATENT) \
	# end of list

PUBS=\
	$(OPATH_CREATE) \
	$(OPATH_PATENT) \
	# end of list


input: $(INPUTS)
build: $(OUTPUTS)
publish: $(PUBS)

$(DNLD)%:
	curl $(URL)$* > tmp
	zcat tmp | sort -k1,1 -k5,1 -T $(BIGTMP) | gzip > $(DNLD)temp
	rm tmp
	mv $(DNLD)temp $(DNLD)$*

$(TFILES):
	../unichem/unichem.py --base_source $(COLLECTION) --other_sources $(SC_LIST) --refresh

$(CFILES):
	$(foreach CFN, $(CFILE_NAMES), s3cmd get --skip-existing $(S3_DRUGSETS)$(CFN) $(DRUGSETS_DIR)$(CFN);)

$(RFILES): $(CFILES)
	$(foreach SC, $(STARTING_COLLECTIONS), tail -n +2 $(DRUGSETS_DIR)$(CFILE_PREF)$(SC)$(CFILE_SUF) | cut -f 1 | sort -u >  $(SC)$(RFILE_SUF);)

$(OPF_PREF)%: $(INPUTS)
	python $(PRS_PRGM) $(DNLD)$* $(TF_LIST) $(RF_LIST) -drug_out temp1 -patent_out temp2
	mv temp1 $(OCF_PREF)$*
	mv temp2 $(OPF_PREF)$*

$(OUTPUTS): $(IOPS_CREATE)
	cat $(OCF_PREF)* | gzip > $(OFILE_CREATE)
	cat $(OPF_PREF)* | gzip > $(OFILE_PATENT)

$(PUBS): $(OUTPUTS)
	cp $(OFILE_CREATE) $(OPATH_CREATE)
	cp $(OFILE_PATENT) $(OPATH_PATENT)

publish_s3: $(PUBS)
	s3cmd put $(OPATH_PATENT) $(S3_STORAGE)$(OFILE_PATENT)
	s3cmd put $(OPATH_CREATE) $(S3_STORAGE)$(OFILE_CREATE)

clean:
	rm -rf *.tsv *.txt *.log *gz
