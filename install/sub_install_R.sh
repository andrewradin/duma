#!/bin/bash

set -e # Halt on error

# This runs after py3 setup, so assume conda exists already.
CONDADIR="$HOME/2xar/opt/conda"
. $CONDADIR/etc/profile.d/conda.sh

CONDA=conda


RENV="r-env"
RENV_DIR="${CONDADIR}/envs/${RENV}"

if [ ! -d "$CONDADIR/envs/$RENV" ]; then
    $CONDA create --yes -n $RENV 
fi


# Activate the R environment
$CONDA activate $RENV

$CONDA config --env --set channel_priority strict
$CONDA config --env --add channels defaults
$CONDA config --env --add channels bioconda
$CONDA config --env --add channels conda-forge

BIOCS=(
      "bioconductor-agilp"
      "bioconductor-arrayexpress"
      "bioconductor-arrayqualitymetrics"
      "bioconductor-biobase"
      "bioconductor-convert"
      "bioconductor-edger"
      "bioconductor-geoquery"
      "bioconductor-hgfocuscdf"
      "bioconductor-hgu133acdf"
      "bioconductor-hgu133b.db"
      "bioconductor-hgu133bcdf"
      "bioconductor-hgu133plus2.db"
      "bioconductor-hgu95av2cdf"
      "bioconductor-hu6800cdf"
      "bioconductor-huex10stprobeset.db"
      "bioconductor-hugene10sttranscriptcluster.db"
      "bioconductor-hugene10stv1cdf"
      "bioconductor-impute"
      "bioconductor-limma"
      "bioconductor-lumi"
      "bioconductor-oligo"
      "bioconductor-org.hs.eg.db" # human
      "bioconductor-org.mm.eg.db" # mouse
      "bioconductor-org.rn.eg.db" # rat
      "bioconductor-org.cf.eg.db" # dog
      "bioconductor-org.dr.eg.db" # zebrafish
      "bioconductor-pcamethods"
      "bioconductor-pd.clariom.d.human"
      "bioconductor-pd.hg.focus"
      "bioconductor-pd.hg.u133.plus.2"
      "bioconductor-pd.hg.u133a"
      "bioconductor-pd.hg.u133a.2"
      "bioconductor-pd.hg.u133b"
      "bioconductor-pd.hg.u95a"
      "bioconductor-pd.ht.hg.u133a"
      "bioconductor-pd.hta.2.0"
      "bioconductor-pd.huex.1.0.st.v2"
      "bioconductor-pd.hugene.1.0.st.v1"
      "bioconductor-pd.hugene.1.1.st.v1"
      "bioconductor-pd.mirna.4.0"
      "bioconductor-pd.u133.x3p"
      "bioconductor-rankprod"
      "bioconductor-sradb"
      "bioconductor-sva"
      "bioconductor-tximport"
      "bioconductor-u133x3pcdf"
      "bioconductor-uniprot.ws"
	)

OTHER=(
         "glib=2.56" # This seems to fix some binary issues with oligo
         "r-base=3.5"
         "r-caret"
         "r-dbi"
         "r-devtools"
         "r-dplyr"
         "r-e1071"
         "r-evaluate"
         "r-extremes"
         "r-git2r"
         "r-httr"
         "r-jsonlite"
         "r-memoise"
         "r-optparse"
         "r-pheatmap"
         "r-quantreg"
         "r-rcpp"
         "r-rcpparmadillo"
         "r-readr"
         "r-rgeos"
         "r-rjson"
         "r-rmtstat"
         "r-rmysql"
         "r-robustbase"
         "r-rocr"
         "r-rook"
         "r-roxygen2"
         "r-rstudioapi"
         "r-rversions"
         "r-samr"
         "r-scales"
         "r-tidyverse"
         "r-venndiagram"
         "r-whisker"
         "r-xml2"
	)


$CONDA install -S --yes ${BIOCS[@]} ${OTHER[@]}

# This should be fixed soon upstream, but for now there are some paths
# we need to override.
cp ${RENV_DIR}/lib/R/etc/Renviron ~/.Renviron
echo -e "R_GZIPCMD=$(which gzip)\nTAR=$(which tar)" >> ~/.Renviron


# We need the development version of scde, which isn't available in conda.
# Additionally, it depends on a specific old version of flexmix that doesn't
# have a binary build available in conda for the right R version.
# So we install both into the environment via devtools.
Rscript $(dirname "$0")/sub_install_scde.R

# Stick our version of R into the py3web1 environment.
# This isn't the cleanest way to handle things, but it is more convenient to
# be able to call everything from that environment.
PY3ENV_DIR="${CONDADIR}/envs/py3web1"
if [ ! -f "${PY3ENV_DIR}/bin/R" ]; then
    ln -s ${RENV_DIR}/bin/R ${PY3ENV_DIR}/bin/R
    ln -s ${RENV_DIR}/bin/Rscript ${PY3ENV_DIR}/bin/Rscript
fi


# Deactivate the R environment
$CONDA deactivate
