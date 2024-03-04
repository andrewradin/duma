default:
	@echo "USAGE: make input|build|clean"

include ../version_tools/setup.mk
include versions.mk

FILE_CLASS=ukbb
PRS_PRGM=parse_ukbb.py
OFILE_STEM=$(FILE_CLASS).$(OUT_VER)


PARSE_MANIFEST=manifest2download.py
URL=https://www.dropbox.com/s/d9ovwwy6wqwjwru/20001_1001.gwas.imputed_$(UKBB_INPT_VER).both_sexes.tsv.bgz
S3_STORAGE=s3://duma-datasets/
DNLD=$(shell ../../web1/path_helper.py downloads)
#downloaded from https://docs.google.com/spreadsheets/d/1kvPoupSzsSFBNSztMzl04xMoSC3Kcx3CrjVf4yBmESU/edit?ts=5b5f17db#gid=178908679
MANIFEST_DOC=$(DNLD)ukbb_gwas_imputed$(UKBB_INPT_VER).file_manifest_release_20180731.manifest_201807.tsv
# hand annotated
WHITE_LIST=$(DNLD)ukbb_whitelist.tsv
UKBB_DNLD=$(DNLD)$(OFILE_STEM)/
OSNPS_UNFILTERED=$(OFILE_STEM).data.tsv.gz
OSNPS=$(OFILE_STEM).data_filtered.tsv.gz
OSTUDIES=$(OFILE_STEM).studies.tsv

ARCHIVED_INPUTS=$(MANIFEST_DOC) $(WHITE_LIST)

show_downloads:
	@echo $(INPUTS)

$(ARCHIVED_INPUTS):
	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_prompt


INPUTS=\
	$(UKBB_DNLD)Z87.gwas.imputed_$(UKBB_INPT_VER).both_sexes.tsv.bgz \
	# there are actually a lot of inputs
	# this is just the last one that should be downloaded

OUTPUTS=\
	$(OSNPS) \
	$(OSTUDIES) \
	# end of list

input: $(INPUTS)
build: $(OUTPUTS)

$(INPUTS): $(ARCHIVED_INPUTS)
	# grab the studies we want from the manifest
	python $(PARSE_MANIFEST) -i $(MANIFEST_DOC) -o todownload_possibledups.tmp
	sort -u todownload_possibledups.tmp > temp
	mv temp todownload.tmp
	rm todownload_possibledups.tmp
	# The files are too large to all store locally, filter down to p<.01
	-mkdir $(UKBB_DNLD)
	awk 'BEGIN{OFS="\n";} NR>1{print $$5,$$4}' todownload.tmp | \
	parallel --max-args=2 "wget {1} -O - | zcat | grep -v 'e-01$$' | grep -v 'e-02$$' | grep -v 'e-03$$' | grep -v 'e-04$$' | grep -v 'NaN$$' | bgzip > $(UKBB_DNLD)/{2}"

$(OSNPS_UNFILTERED) $(OSTUDIES): $(INPUTS)
	echo -e "Phenotype|PMID \tTotalSamples(discovery+replication)\tGWASancestryDescription Platform [SNPs passing QC]\tDatePub\tIncludesMale/Female Only Analyses\tExclusively Male/Female European Discovery\tAfrican Discovery\tEast Asian Discovery\tEuropean Replication\tAfrican Replication\tEast Asian Replication" > outstudies.tmp
	echo -e "phenotype|study\trsid\tchr\tpos\tpval\tmaf\tflag" > outsnps.tmp
	gzip -f outsnps.tmp
# for testing use the first line, but switch back for production
#	tail -n +2 todownload.tmp | head -3 > nhtodownload.tmp
	tail -n +2 todownload.tmp > nhtodownload.tmp
	while read p; do \
	echo $$p | \
	(read p1 p2 p3 p4 p5; \
	python parse_ukbb.py \
	 -i $(UKBB_DNLD)$$p4 \
	 -p 0.05 -study "ukbb_gwas_imputedv3" \
	 -pheno $$p2 \
	 -sex $$p3 \
	 -osnp outsnps.tmp.gz \
	 -ostudy outstudies.tmp) \
	done < nhtodownload.tmp
	mv outsnps.tmp.gz $(OSNPS_UNFILTERED)
	mv outstudies.tmp $(OSTUDIES)

$(OSNPS): $(OSNPS_UNFILTERED)
	# -i for case insensitive; the whitelist includes casing (though I suspect some of it is wrong), but
	# since we are now case insensitive, need to match regardless of case.
	zgrep -i -F -f $(WHITE_LIST) $(OSNPS_UNFILTERED) | gzip > $(OSNPS).tmp
	mv $(OSNPS).tmp $(OSNPS)


clean:
	rm -rf *.log *.gz *.tsv *.tmp
