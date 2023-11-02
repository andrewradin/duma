default:
	@echo "USAGE: make build|publish|publish_s3|clean"

FILE_CLASS=unichem

include ../version_tools/setup.mk
include versions.mk

# Since the output files are, for the most part, just renamed download files,
# this directory doesn't have a separate 'input' makestep.

# It's hard to tell from the website what has changed. The easiest refresh
# process is to
# - define a new version with the current date in versions.py
# - run 'make build'
# - examine the outputs and, if they're not sufficiently different, delete
#   them and back out the versions.py change.

PRS_PRGM=./unichem.py

OUT_PAT=$(FILE_CLASS).$(OUT_VER).%.tsv.gz
OUTPUTS=$(patsubst %,$(OUT_PAT),$(MAPPINGS))

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo UNICHEM_VER=`date +%Y-%m-%d`
	@echo
	@echo 'Then:'
	@echo make build
	@echo manually verify changes are worth publishing
	@echo '(if not, stop and revert versions.py changes)'
	@echo make publish_s3
	@echo
	@echo 'Note: there is no input makestep because the download happens'
	@echo 'on the fly during "make build"'

$(OUT_PAT):
	[ '$(UNICHEM_VER)' = `date +%Y-%m-%d` ] # verify configured date
	$(PRS_PRGM) --pattern $(OUT_PAT) --mapping $* --refresh

build: $(OUTPUTS)

clean:
	rm -rf *.tsv *.txt *.log *gz
