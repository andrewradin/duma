default:
	@echo "USAGE: make show_latest_version|build|clean"

FILE_CLASS=globaldata

include ../version_tools/setup.mk
include versions.mk

ATTR_FILE=$(FILE_CLASS).full.$(OUT_VER).attributes.tsv
DPI_FILE=$(FILE_CLASS).default.$(OUT_VER).evidence.tsv
PARSE_FILE=$(FILE_CLASS).log.$(OUT_VER).parse_detail.tsv

# temporary files
TMP_ATTR=tmp.attributes.tsv
TMP_DPI=tmp.evidence.tsv
TMP_PARSE=parse_errors.tsv
TMP_CLUST=clusters.tsv

UNIPROT_CHOICE=HUMAN_9606.v$(UNIPROT_VER)

OUTPUTS=\
	$(ATTR_FILE) \
	$(DPI_FILE) \
	$(PARSE_FILE) \
	# end of list

show_latest_version:
	@echo Update versions.py with:
	@echo GLOBALDATA_VER=`date +'%Y-%m-%d'`
	@echo and set the desired UNIPROT_VER
	@echo
	@echo Make sure your key won\'t be invalidated, by running:
	@echo make refresh_key
	@echo
	@echo Then run:
	@echo make build
	@echo make publish_s3
	@echo
	@echo Note that this will load the selected UNIPROT_VER
	@echo "into the local database, if it isn't there already."

# by forcing a refresh here, we assure that the cron background refresh
# won't happen for at least 3 weeks, so the key should stay valid
# throughout the extraction process
refresh_key:
	../../web1/scripts/apikeys.py --renew --force GD

build: $(OUTPUTS)

clean:
	-rm *.tsv

# XXX With deduplication and all synonyms stripped, matching runs to completion
# XXX and results are similar to before (see:
# XXX vimdiff without-gd-logdir/match.log log/2022.06.18-11:07:30/match.log
# XXX ), but 80% of globaldata molecules don't cluster:
# XXX - 51421 total global data keys
# XXX - 39450 completely disconnected
# XXX - 11971 in 2nd-round clustering
# XXX - clusters went from 497913 to 499019 (+1106)
# XXX - clustered drugs went from 1153201 to 1166221 (+13020)
# XXX - after cluster breaking:
# XXX   - clusters went from 502517 to 503694 (+1177)
# XXX   - clustered drugs went from 1152846 to 1165583 (+12737)
# XXX - 11659 globaldata drugs have CAS, so probably the ones that don't are
# XXX   what ends up being disconnected
# XXX - in the original globaldata file, 8210 distinct cas numbers appear with
# XXX   137199 synonyms
# XXX After more agressive input filtering, we have:
# XXX - 35175 total global data keys
# XXX - 11284 in clusters
# XXX - 23891 disconnected
# XXX After some extraction tweaks on 2022-12-09, relative to v3:
# XXX - 36132 -> 29722 total global data keys
# XXX - 11550 -> 11545 in clusters
# XXX - 24270 -> 17819 disconnected
# XXX ...so, this extracted fewer useless drugs, but didn't help clustering

# In the following, pull_drugs downloads the GD drug data via the API,
# and creates initial attr and dpi files (with hardcoded names, corresponding
# to $(TMP_ATTR) and $(TMP_DPI)).
# The first run of deduplicate reads the attr file and groups together
# GD drugs with the same canonical name, and writes that mapping to
# $(TMP_CLUST).
# The second run of deduplicate combines all attr data for GD drugs in the
# same cluster into a single drug, recording the other GD ids in the cluster
# as shadowed ids.
# The third run of deduplicate combines all dpi data for GD drugs in the
# same cluster into a single drug.
#
# XXX Since the pull step is slow, and deduplication is fast, for testing
# XXX you might want to run the deduplicate steps manually against old data,
# XXX before creating a new version. Also, you might prefer adding
# XXX post-processing logic in deduplicate, rather than in pull_drugs.
$(OUTPUTS):
	[ '$(GLOBALDATA_VER)' = `date +%Y-%m-%d` ] # verify configured date
	./pull_drugs.py --load-uniprot $(UNIPROT_CHOICE)
	./deduplicate.py canonical $(TMP_ATTR) $(TMP_CLUST)
	./deduplicate.py --clusters $(TMP_CLUST) attr $(TMP_ATTR) $(ATTR_FILE)
	./deduplicate.py --clusters $(TMP_CLUST) dpi $(TMP_DPI) $(DPI_FILE)
	mv parse_errors.tsv $(PARSE_FILE)
