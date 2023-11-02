default:
	@echo "USAGE: make input|build|clean"

FILE_CLASS=salmon

include ../version_tools/setup.mk
include versions.mk

DNLD=$(shell ../../web1/path_helper.py downloads)
PYTHON_RUNNER=./salmon_index_runner.py
WORKER_NAME=worker-test

BASE_ENSEMBL_URL=ftp://ftp.ensembl.org/pub/release-$(ENSEMBL_VER)/fasta/

H_ENSEMBL_URL=$(BASE_ENSEMBL_URL)homo_sapiens/cdna/
H_URL_FILE=Homo_sapiens.GRCh$(HG_V).cdna.all.fa.gz
H_XX_GZIP=$(FILE_CLASS).ensembl$(ENSEMBL_VER).$(H_URL_FILE)
H_XX=$(DNLD)$(H_XX_GZIP)

M_ENSEMBL_URL=$(BASE_ENSEMBL_URL)mus_musculus/cdna/
M_URL_FILE=Mus_musculus.$(MM_V).cdna.all.fa.gz
M_XX_GZIP=$(FILE_CLASS).ensembl$(ENSEMBL_VER).$(M_URL_FILE)
M_XX=$(DNLD)$(M_XX_GZIP)

R_ENSEMBL_URL=$(BASE_ENSEMBL_URL)rattus_norvegicus/cdna/
R_URL_FILE=Rattus_norvegicus.$(RN_V).cdna.all.fa.gz
R_XX_GZIP=$(FILE_CLASS).ensembl$(ENSEMBL_VER).$(R_URL_FILE)
R_XX=$(DNLD)$(R_XX_GZIP)

D_ENSEMBL_URL=$(BASE_ENSEMBL_URL)canis_lupus_familiaris/cdna/
D_URL_FILE=Canis_lupus_familiaris.$(CF_V).cdna.all.fa.gz
D_XX_GZIP=$(FILE_CLASS).ensembl$(ENSEMBL_VER).$(D_URL_FILE)
D_XX=$(DNLD)$(D_XX_GZIP)

Z_ENSEMBL_URL=$(BASE_ENSEMBL_URL)danio_rerio/cdna/
Z_URL_FILE=Danio_rerio.$(DR_V).cdna.all.fa.gz
Z_XX_GZIP=$(FILE_CLASS).ensembl$(ENSEMBL_VER).$(Z_URL_FILE)
Z_XX=$(DNLD)$(Z_XX_GZIP)

show_downloads:
	@echo $(H_ENSEMBL_URL)$(H_URL_FILE)
	@echo $(M_ENSEMBL_URL)$(M_URL_FILE)
	@echo $(R_ENSEMBL_URL)$(R_URL_FILE)
	@echo $(D_ENSEMBL_URL)$(D_URL_FILE)
	@echo $(Z_ENSEMBL_URL)$(Z_URL_FILE)

INPUTS=\
	$(H_XX) \
	$(M_XX) \
	$(R_XX) \
	$(D_XX) \
	$(Z_XX) \
	# end of list

$(H_XX):
	wget -O - $(H_ENSEMBL_URL)$(H_URL_FILE) > temp
	mv temp $(H_XX)
$(M_XX):
	wget -O - $(M_ENSEMBL_URL)$(M_URL_FILE) > temp
	mv temp $(M_XX)
$(R_XX):
	wget -O - $(R_ENSEMBL_URL)$(R_URL_FILE) > temp
	mv temp $(R_XX)
$(D_XX):
	wget -O - $(D_ENSEMBL_URL)$(D_URL_FILE) > temp
	mv temp $(D_XX)
$(Z_XX):
	wget -O - $(Z_ENSEMBL_URL)$(Z_URL_FILE) > temp
	mv temp $(Z_XX)

# leaving the human species unlabeled as such
# mostly for back compatability
OFILE31=$(FILE_CLASS).human.$(OUT_VER).kmer31.tgz
OFILE23=$(FILE_CLASS).human.$(OUT_VER).kmer23.tgz

# same file types for the model organisms
M_OFILE31=$(FILE_CLASS).mouse.$(OUT_VER).kmer31.tgz
M_OFILE23=$(FILE_CLASS).mouse.$(OUT_VER).kmer23.tgz
R_OFILE31=$(FILE_CLASS).rat.$(OUT_VER).kmer31.tgz
R_OFILE23=$(FILE_CLASS).rat.$(OUT_VER).kmer23.tgz
D_OFILE31=$(FILE_CLASS).dog.$(OUT_VER).kmer31.tgz
D_OFILE23=$(FILE_CLASS).dog.$(OUT_VER).kmer23.tgz
Z_OFILE31=$(FILE_CLASS).zebrafish.$(OUT_VER).kmer31.tgz
Z_OFILE23=$(FILE_CLASS).zebrafish.$(OUT_VER).kmer23.tgz

OUTPUTS=\
	$(OFILE31) \
	$(OFILE23) \
	$(M_OFILE31) \
	$(M_OFILE23) \
	$(R_OFILE31) \
	$(R_OFILE23) \
	$(D_OFILE31) \
	$(D_OFILE23) \
	$(Z_OFILE31) \
	$(Z_OFILE23) \
	# end of list

# XXX This could pretty easily be cleaned up with some snazzier Makefile code, but this works...
$(OFILE31): $(H_XX)
	$(PYTHON_RUNNER) --kmer 31 --cdna $(H_XX) -o $(OFILE31) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(OFILE23): $(H_XX)
	$(PYTHON_RUNNER) --kmer 23 --cdna $(H_XX) -o $(OFILE23) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(M_OFILE31): $(M_XX)
	$(PYTHON_RUNNER) --kmer 31 --cdna $(M_XX) -o $(M_OFILE31) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(M_OFILE23): $(M_XX)
	$(PYTHON_RUNNER) --kmer 23 --cdna $(M_XX) -o $(M_OFILE23) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(R_OFILE31): $(R_XX)
	$(PYTHON_RUNNER) --kmer 31 --cdna $(R_XX) -o $(R_OFILE31) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(R_OFILE23): $(R_XX)
	$(PYTHON_RUNNER) --kmer 23 --cdna $(R_XX) -o $(R_OFILE23) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(D_OFILE31): $(D_XX)
	$(PYTHON_RUNNER) --kmer 31 --cdna $(D_XX) -o $(D_OFILE31) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(D_OFILE23): $(D_XX)
	$(PYTHON_RUNNER) --kmer 23 --cdna $(D_XX) -o $(D_OFILE23) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(Z_OFILE31): $(Z_XX)
	$(PYTHON_RUNNER) --kmer 31 --cdna $(Z_XX) -o $(Z_OFILE31) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

$(Z_OFILE23): $(Z_XX)
	$(PYTHON_RUNNER) --kmer 23 --cdna $(Z_XX) -o $(Z_OFILE23) -r $(WORKER_NAME) --pkg_version $(SALMON_PKG_V)

input: $(INPUTS)
build: $(OUTPUTS)

show_latest_version:
	@echo
	@echo '======================================'
	@echo 'versions.py entry should contain:'
	@echo 'The latest version of the UCSC human genome build.'
	@echo '  To verify that, update UCSC_HG, as needed, and update X in the'
	@echo '  following lines with our latest version number for UCSC_HG'
	@echo '  UCSC_HG_VER=X,'
	@echo '  HG_V=get_hg_v(X)'
	@echo
	@echo 'It should also contain the latest ENSEMBL genome build.'
	@echo '  Go to http://ftp.ensembl.org/pub/ and look for latest release-XXX directory.'
	@echo "  Then update ENSEMBL_VER='XXX'"
	@echo
	@echo 'Also update the model organism versions.'
	@echo 'Appropriate versions can be searched for here:'
	@echo ' https://uswest.ensembl.org/info/about/species.html'
	@echo
	@echo 'Also compare these values to those used for the homologene ETL. They need to be the same'
	@echo
	@echo "This is also a good time to update the Salmon install to the latest version."
	@echo " The package version is tracked in versions.py"
	@echo "Updating the packge is done by updating the version name in"
	@echo " web1/R/RNAseq/Makefile and running that updated Makefile on $(WORKER_NAME)"
	@echo
	@echo Then:
	@echo make input
	@echo make build
	@echo make publish_s3


clean:
	rm -rf ./tmp

