# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# #find edges that are consistently higher than 0

# %% vscode={"languageId": "r"}
renv::load()

# %% vscode={"languageId": "r"}
library('tidyverse') 
library('psych')
library('glue') 
library('ini')
config <- read.ini("../../../config.ini")



# %% [markdown]
# ### MSMAll_FIX
#

# %% vscode={"languageId": "r"}
proc_type <- 'MSMAll_FIX'
results_path  <- file.path(config$Paths$script_dir, 'results')
plotting_path <- file.path(config$Paths$script_dir, 'results', 'plotting')

# %% vscode={"languageId": "r"}
lasso_fix_df <- read.csv(file.path(results_path, 'MSMAll-FIX', 'proc-FIX_model-lassoBIC_reliability-edges.csv')) 
pearson_fix_df <- read.csv(file.path(results_path, 'MSMAll-FIX', 'proc-FIX_model-Pearson_reliability-edges.csv')) 
uoi_fix_df <- read.csv(file.path(results_path, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_reliability-edges.csv')) 
ridge_fix_df <- read.csv(file.path(results_path, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_reliability-edges.csv')) 
pcorr_fix_df <- read.csv(file.path(results_path, 'MSMAll-FIX', 'proc-FIX_model-partial-correlation_reliability-edges.csv')) 

# %% [markdown]
# #### ICC 

# %% vscode={"languageId": "r"}
edge_icc <- function(df) { 
    df <- df %>% dplyr::select(c(sub, values, ses) ) %>% pivot_wider(names_from = ses, 
                                                          values_from = values) %>% dplyr::select(c('ses-1', 'ses-2'))

    icc2 <- psych::ICC(df)$results['Average_random_raters', 'ICC']
    return(icc2)
}

# %% vscode={"languageId": "r"}
lasso_split_df <- split(lasso_fix_df, lasso_fix_df$pos)

lasso_fix_results <- lapply(lasso_split_df, function(df) {
  edge_icc(df)
})
write.csv(lasso_fix_results,file=glue("{plotting_path}/MSMAll-FIX/proc-FIX_model-lassoBIC_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
pearson_split_df <- split(pearson_fix_df, pearson_fix_df$pos)

pearson_fix_results <- lapply(pearson_split_df, function(df) {
  edge_icc(df)
})
write.csv(pearson_fix_results,file=glue("{plotting_path}/MSMAll-FIX/proc-FIX_model-Pearson_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
ridge_split_df <- split(ridge_fix_df, ridge_fix_df$pos)

ridge_fix_results <- lapply(ridge_split_df, function(df) {
  edge_icc(df)
})

write.csv(ridge_fix_results,file=glue("{plotting_path}/MSMAll-FIX/proc-FIX_model-ridgeCV_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
uoi_split_df <- split(uoi_fix_df, uoi_fix_df$pos)

uoi_fix_results <- lapply(uoi_split_df, function(df) {
  edge_icc(df)
})

write.csv(uoi_fix_results,file=glue("{plotting_path}/MSMAll-FIX/proc-FIX_model-uoiLasso_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
pcorr_split_df <- split(pcorr_fix_df, pcorr_fix_df$pos)

pcorr_fix_results <- lapply(pcorr_split_df, function(df) {
  edge_icc(df)
})
write.csv(pcorr_fix_results,file=glue("{plotting_path}/MSMAll-FIX/proc-FIX_model-partial-correlation_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% [markdown]
# ### XCPD

# %% vscode={"languageId": "r"}
lasso_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-lassoBIC_reliability-edges.csv'))
pearson_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-Pearson_reliability-edges.csv')) 
uoi_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-uoiLasso_reliability-edges.csv')) 
ridge_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-ridgeCV_reliability-edges.csv')) 
pcorr_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-partial-correlation_reliability-edges.csv')) 


# %% vscode={"languageId": "r"}
split_df <- split(lasso_xcpd_df, lasso_xcpd_df$pos)

lasso_xcpd_results <- lapply(split_df, function(df) {
  edge_icc(df)

})

write.csv(lasso_xcpd_results,file=glue("{plotting_path}/XCPD/proc-XCPD_model-lassoBIC_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
split_df <- split(pearson_xcpd_df, pearson_xcpd_df$pos)

pearson_xcpd_results <- lapply(split_df, function(df) {
  edge_icc(df)
})

write.csv(pearson_xcpd_results,file=glue("{plotting_path}/XCPD/proc-XCPD_model-Pearson_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
split_df <- split(ridge_xcpd_df, ridge_xcpd_df$pos)

ridge_xcpd_results <- lapply(split_df, function(df) {
  edge_icc(df)
})

write.csv(ridge_xcpd_results,file=glue("{plotting_path}/XCPD/proc-XCPD_model-ridgeCV_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
split_df <- split(uoi_xcpd_df, uoi_xcpd_df$pos)

uoi_xcpd_results <- lapply(split_df, function(df) {
  edge_icc(df)
})

write.csv(uoi_xcpd_results,file=glue("{plotting_path}/XCPD/proc-XCPD_model-uoiLasso_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)

# %% vscode={"languageId": "r"}
split_df <- split(pcorr_xcpd_df, pcorr_xcpd_df$pos)

pcorr_xcpd_results <- lapply(split_df, function(df) {
  edge_icc(df)
})

write.csv(pcorr_xcpd_results,file=glue("{plotting_path}/XCPD/proc-XCPD_model-partial-correlation_stat-ICC.csv"), row.names=FALSE, col.names=FALSE)
