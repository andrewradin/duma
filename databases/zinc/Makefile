default:
	@echo "USAGE: make build|publish|clean"

FILE_CLASS=zinc

include ../version_tools/setup.mk
include versions.mk

URL_PRE=https://zinc20.docking.org/substances/subsets/
URL_SUF=.csv:zinc_id?count=all

OPREF=$(FILE_CLASS).$(OUT_VER).
OSUF=.tsv.gz

SUBSETS=$(shell python ../../web1/dtk/zinc.py --get_labels)

OUTPUTS=$(foreach I, $(SUBSETS), $(OPREF)$(I)$(OSUF))

test:
	echo $(OUTPUTS)

build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo ZINC_VER=`date +%Y-%m-%d`
	@echo
	@echo 'Then:'
	@echo make build
	@echo manually verify changes are worth publishing
	@echo '(if not, stop and revert versions.py changes)'
	@echo make publish_s3
	@echo
	@echo 'Note: there is no input makestep because the download happens'
	@echo 'on the fly during "make build"'


$(OPREF)%$(OSUF):
	[ '$(ZINC_VER)' = `date +%Y-%m-%d` ] # verify configured date
	-rm tmp
	wget -O tmp $(URL_PRE)$*$(URL_SUF)
	cat tmp | gzip > $(OPREF)$*$(OSUF)

clean:
	rm -rf *.tsv *.tsv *.log *gz tmp
