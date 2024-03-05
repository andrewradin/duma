library(devtools)
if (!require(flexmix) || packageVersion("flexmix") != "2.3.13") {
    devtools::install_version('flexmix', '2.3-13', repos='https://cloud.r-project.org', build_vignettes=FALSE, upgrade_dependencies=FALSE)
}
devtools::install_github('hms-dbmi/scde', build_vignettes=FALSE, dependencies=FALSE)
