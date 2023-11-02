default:
	@echo "USAGE: make build|publish_ws|publish_s3|clean"

PRCS_PRGM=../../web1/scripts/connect_drugs_to_proteinSets.py
ODIR=$(shell ../../web1/path_helper.py storage)d2ps
OUT_SUF=.tsv.gz

DPIS=\
	dpimerge.default \
	dpimerge.chembl \
	dpimerge.DNChB \
	dpimerge.DNChB_ki \
	dpimerge.DNChBX_ki \
	dpimerge.chembl_homologous \
	drugbank.default \
	# end of list

PPIS=\
	string.default \
	drpias.default \
	# end of list

GMTS=\
	reactome \
	# end of list

OUTPUTS=$(foreach dpi, $(DPIS),$(foreach ppi, $(PPIS),$(foreach gmt, $(GMTS),$(dpi)_$(ppi)_$(gmt)$(OUT_SUF))))
PUBS=$(foreach o, $(OUTPUTS),$(ODIR)/$(o))

# This makefile gets invoked automatically from the 'matching' directory to
# rebuild d2ps files when the dpi files change. Since this can be expensive
# (chembl builds take close to an hour each), we try to avoid work where
# possible:
# - this directory only rebuilds a file if it's out of date with respect to
#   the corresponding dpi file in ws/dpi
# - the matching directory only copies a file in to ws/dpi if its content is
#   different from what's already there
# This means the matching directory Makefile can invoke publish_ws directly
# after publishing to ws/dpi, and again implicitly when publishing to S3,
# without duplicating any builds.

# for the build step, invoke make recursively for each output; the DPI, PPI
# and GMT variables being set allows a single make rule to generate the
# correct build params and dependencies for each combination

build_one: $(DPI)_$(PPI)_$(GMT)$(OUT_SUF)

build:
	@for DPI in $(DPIS) ;\
	do \
		for PPI in $(PPIS) ;\
		do \
			for GMT in $(GMTS) ;\
			do \
				$(MAKE) DPI=$$DPI PPI=$$PPI GMT=$$GMT build_one ;\
			done; \
		done; \
	done

DPI_SRC=$(shell ../../web1/path_helper.py storage)dpi/dpi.$(DPI).tsv
# rebuild a single output only if it's missing or its inputs have changed
# XXX can extend with PPI_SRC, GMT_SRC as these are automated
$(DPI)_$(PPI)_$(GMT)$(OUT_SUF): $(DPI_SRC)
	$(PRCS_PRGM) --min_ppi_evid 0.9 --dpi $(DPI) --ppi $(PPI) --gmt $(GMT)

# publish_ws copies any files changed since the last publish
publish_ws: $(PUBS)

$(ODIR)/%:%
	cp $^ $@

# publish_s3 makes sure everything is up to date, and copies everything
# in ws to S3
publish_s3: build publish_ws
	@for FILE in $(OUTPUTS) ;\
	do \
		../matching/move_s3_files.py --put d2ps $$FILE ;\
	done
