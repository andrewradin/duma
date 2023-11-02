# To use versioning tools, an ETL directory should:
# - add these two lines to the Makefile:
#     include ../version_tools/setup.mk
#     include versions.mk
# - add a (checked-in) versions.py file that defines the Make variables used
#   to make each version. These should define both the (external) version of
#   the data being imported, and the (duma) version of any other ETL outputs
#   used in the import process. A versions.py file looks like:
#	  versions = {
#		  1:dict(VAR1='value1',VAR2='value2'),
#		  2:dict(VAR1='value1',VAR2='value2'),
#		  }
#
# The Makefile will then have all the variables defined by the versions.py
# file, plus:
#   OUT_VER_NBR - contains the highest key in versions.py
#   OUT_VER - the above with a 'v' prefixed; should be used
#             as the version part for the output filename(s)
#   PUB_VER_FILE - the name of a local file for use in tracking when a
#             version is published
#   PUBLISHED - which will only be defined if OUT_VER has already been
#             published to S3
#
# The Makefile in the individual ETL directory should define:
#   OUTPUTS - the list of output files generated
#   FILE_CLASS - the virtual bucket files are published into
#   'input' and 'build' targets
# Given the above, this file will provide:
#   the variables:
#     PUBDIR - the full path to the directory where files are published
#   and the targets:
#     show_outputs
#     publish
#     publish_s3
#     and an implicit rule for publishing a single file
#
# Note that for input files being retrieved from a remote database, the
# makefile will need to manage two distinct names:
# - the name of the file in the URL as it is being fetched, which we have
#   no control over
# - the name of the file in the ws/downloads directory, which should:
#   - begin with the FILE_CLASS, to avoid conflicts with other ETL directories
#     that all share ws/downloads
#   - contain the source version, to distinguish it from any previous releases
#     that have been downloaded
# Previously, ETL logic tended to re-use the URL filename in the downloads
# directory, and depend on luck to avoid conflicts.
#
# If the ETL Makefile also defines ARCHIVED_INPUTS (as a list of paths of files
# in ws/downloads) then these downloaded inputs will be backed up on S3 in a
# standard way as part of the publish_s3 set.
#
# If using ARCHIVED_INPUTS, and the ETL Makefile includes:
#
# $(ARCHIVED_INPUTS):
#	$(MAKE) DOWNLOAD_FILE=$(notdir $@) get_s3_input_or_download
#
# download_input:
#   whatever is needed to download the current version
#
# then the files will be retrieved from S3 if available, and will be downloaded
# from the source if not. Note that, since the use case is one where a
# particular version is available from the source for a limited time, the
# actual download case should typically happen once, right after editing
# versions.py.
#
# If an automatic download is difficult, or it's difficult to confirm the
# file version, the target get_s3_input_or_prompt may be substituted. This
# will produce an error message if the file isn't on S3, and request a
# manual download.
#
# A Makefile can also use a more customized retrieval; see uniprot as an
# example.

# The expected use pattern for publishing a new version is:
# - run 'make show_latest_version'. This will either output definitions to
#   be added to versions.py for defining a new version, or instructions for
#   manually retrieving that information. This should also output any other
#   special instructions or considerations for the person doing the refresh.
# - add the description to versions.py
# - run the input, build, and publish make targets, testing results and
#   making code changes as necessary
# - when ready to release, run the publish_s3 make target, which will
#   also update the local file last_published_version
# - commit the updates to versions.py and last_published_version, along
#   with any code updates
#
# If the above procedure is followed, the git log of changes to
# last_published_version will provide the commits of the code used
# to create each version.


# == Setup some defaults to help catch common errors ==

# pipefail ensures that commands of the form:
#    ./run_some_script.sh | gzip | sort > blah.tsv.gz
# will cause the Makefile to halt, instead of continuing
SHELL=/bin/bash -o pipefail

# By default make will happily replace any undefined variables with an empty string
# This makes it warn about doing that.
MAKEFLAGS=--warn-undefined-variables



# == Common values and targets ==
PUB_VER_FILE=last_published_version
VERSION_TOOL=../version_tools/etl_versions.py
STATS_TOOL=../version_tools/stats_tool.py
CP_TOOL=rsync -c
S3_TOOL=../matching/move_s3_files.py

help:
	@echo "USAGE: make input|build|publish|publish_s3|clean"

versions.mk: versions.py $(PUB_VER_FILE)
	pylint -d all -e duplicate-key versions.py
	$(VERSION_TOOL)

WS_DIR=$(shell ../../web1/path_helper.py storage)
PUBDIR=$(WS_DIR)$(FILE_CLASS)

# This step retrieves any input that comes from S3. It uses a new mode in
# move_s3_files that takes a blank bucket name as a signal to parse the
# bucket and file names out of the full cache path of the file. Although
# this would also match a filename to be published, the publish makestep
# below doesn't define individual files as make targets, so there's not
# interference.
$(WS_DIR)%:
	$(S3_TOOL) '' $@

$(PUB_VER_FILE):
	@echo 0 > $(PUB_VER_FILE)

OUT_VER=v$(OUT_VER_NBR)

publish: build
	@mkdir -p $(PUBDIR)
	@for FILE in $(OUTPUTS); do $(CP_TOOL) $$FILE $(PUBDIR); done

# XXX Currently, this doesn't support the gzip option -- files are put
# XXX onto S3 in the same format that they appear under ws. But that
# XXX option is not very well structured in any case, because it requires
# XXX each consumer to know it's in use for the file type, and to call
# XXX S3Bucket.list() and S3File.fetch() with the appropriate parameter.
# XXX Probably, the whole thing should be restructured so that it is
# XXX specified globally which buckets and roles it applies to, and
# XXX this code and the retrieval code could all consult that specification
# XXX without involving the data consumers.
publish_s3: publish
	@test "$(PUBLISHED)" != yes \
		|| (echo THIS VERSION IS ALREADY PUBLISHED && false)
	$(MAKE) archive_inputs
	@for FILE in $(OUTPUTS); do \
		$(S3_TOOL) --verbose --put $(FILE_CLASS) $$FILE; \
	done
	echo $(OUT_VER_NBR) > $(PUB_VER_FILE)

show_outputs:
	@for FILE in $(OUTPUTS); do echo $$FILE; done

show_stats:
	python ./stats.py --version $(OUT_VER_NBR)


archive_inputs:
	@for FILE in $(foreach fn,$(ARCHIVED_INPUTS),$(notdir $(fn))) ;\
	do \
		if ! $(S3_TOOL) --exists \
				--downloads-archive $(FILE_CLASS) $$FILE ;\
		then \
			echo Archiving $$FILE ;\
			$(S3_TOOL) --put \
					--downloads-archive $(FILE_CLASS) $$FILE ;\
		fi ;\
	done

get_s3_input_or_prompt:
	@if $(S3_TOOL) --exists \
			--downloads-archive $(FILE_CLASS) $(DOWNLOAD_FILE);\
	then \
		$(S3_TOOL) --downloads-archive $(FILE_CLASS) $(DOWNLOAD_FILE);\
	else \
		echo ; \
		echo ; \
		echo PLEASE RETRIEVE MISSING INPUT: $(DOWNLOAD_FILE); \
		echo for more information, run "'make show_latest_version'"; \
		echo ; \
		echo ; \
		false ; \
	fi

get_s3_input_or_download:
	@if $(S3_TOOL) --exists \
			--downloads-archive $(FILE_CLASS) $(DOWNLOAD_FILE);\
	then \
		$(S3_TOOL) --downloads-archive $(FILE_CLASS) $(DOWNLOAD_FILE);\
	else \
		$(MAKE) download_input ; \
	fi

statsfile.v%.tsv:
	python ./stats.py --version $* --output $@
