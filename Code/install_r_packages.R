# Install the R packages used by information_content_regressions.R.
# The paper's results were produced with R 4.2.3 and the versions listed below.
#   readxl 1.4.3, lme4 1.1-35.3, lmerTest 3.1-3, performance 0.15.1
#
# To install those exact versions instead of the latest releases, use:
#   install.packages("remotes")
#   remotes::install_version("lme4", version = "1.1-35.3")   # etc.

packages <- c("readxl", "lme4", "lmerTest", "performance")
missing <- packages[!packages %in% rownames(installed.packages())]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
