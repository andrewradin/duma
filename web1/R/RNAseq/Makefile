default:
	@echo "USAGE: make install|clean"

# hardcoded version numbers
TG_V=0.6.6
SALMON_V=1.9.0
SRA_V=3.0.0

STOR=$(shell ../../path_helper.py storage)
DWNLDS=$(shell ../../path_helper.py downloads)

SRA_URL=https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/$(SRA_V)/sratoolkit.$(SRA_V)-ubuntu64.tar.gz
SRA_TGZ=sratoolkit.tar.gz
SRA_TK_DIR=sratoolkit/
FQ_DUMP=$(SRA_TK_DIR)bin/fasterq-dump.$(SRA_V)
### Note that as of v 2.10.* you have to interactively set one parameter
# Run the vdb configuration tool: $(SRA_TK_DIR)bin/vdb-config --interactive.
# Navigate to the AWS tab: press A
# Mark "report cloud instance identity" option with [X]: press r
# Select Save: press s
# Select Exit: press x


TG_ZIP=trim_galore_$(TG_V).zip
TG_URL=https://github.com/FelixKrueger/TrimGalore/archive/$(TG_V).zip
TG_DIR=trim_galore/
TG_V_DIR=$(TG_DIR)/TrimGalore-$(TG_V)/
TG=$(TG_V_DIR)trim_galore

SALMON_URL=https://github.com/COMBINE-lab/salmon/releases/download/v$(SALMON_V)/
SALMON_TMP_DIR=Salmon-$(SALMON_V)_linux_x86_64
SALMON_TGZ=$(SALMON_TMP_DIR).tar.gz
SALMON_DIR=Salmon
SALMON_V_DIR=Salmon$(SALMON_V)
SALMON=$(SALMON_V_DIR)/bin/salmon

ca=ca
fqc=fqc

.PHONY : $(ca)
.PHONY : $(fqc)

PARTS=\
	$(FQ_DUMP) \
	$(TG) \
	$(ca) \
	$(fqc) \
	$(SALMON) \
	# end of list

install: $(PARTS)

$(FQ_DUMP):
	wget -O - $(SRA_URL) > temp
	mv temp $(SRA_TGZ)
	tar -zxvf $(SRA_TGZ)
	rm $(SRA_TGZ)
	rm -rf $(SRA_TK_DIR)
	mv sratoolkit.$(SRA_V)-ubuntu64 $(SRA_TK_DIR)

$(TG):
	wget -O - $(TG_URL) > temp
	mv temp $(TG_ZIP)
	rm -rf $(TG_DIR)
	unzip $(TG_ZIP) -d $(TG_DIR)

$(SALMON):
	rm -rf $(SALMON_DIR)*
	wget -O - $(SALMON_URL)$(SALMON_TGZ) > temp
	mv temp $(SALMON_TGZ)
	mkdir $(SALMON_TMP_DIR)
	tar -zxvf $(SALMON_TGZ) -C $(SALMON_TMP_DIR) --strip-components=1
	rm $(SALMON_TGZ)
	mkdir $(SALMON_DIR)
	mv $(SALMON_TMP_DIR)/* $(SALMON_DIR)/
	rm -rf $(SALMON_TMP_DIR)
	ln -s $(SALMON_DIR) $(SALMON_V_DIR)

# let the pip and apt-get deal with these
$(ca):
	python3 -m pip install cutadapt
$(fqc):
	sudo apt-get install fastqc

clean:
	rm -rf temp
	rm -rf $(SALMON_TGZ)
	rm -rf $(SRA_TGZ)
	rm -rf $(TG_ZIP)
