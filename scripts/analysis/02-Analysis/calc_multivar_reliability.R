# ---
# jupyter:
#   jupytext:
#     formats: ipynb,Rmd,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# #find edges that are consistently higher than 0

# +
# mahadevan implementation: https://github.com/BassettLab/motion-FC-metrics/blob/master/computeICC.m
# -

library('I2C2') 
library('tidyverse') 
library('psychometric')
library('glue') 
# library('psych') 


# +
icc_calc <- function(df) { 

    df$pos <- NULL
        
    wide_df <- pivot_wider(data=df, names_from='X', values_from='values')
        
    y <- wide_df[-1:-2]
    id <- wide_df$sub
    visit <- wide_df$ses
    
    i2c2_res <- I2C2(y, id = id, visit = visit, demean = TRUE) 
    i2c2_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI #na.rm = TRUE if err
    
    icc_obj = list(i2c2_res, i2c2_ci)

return(icc_obj) 
 } 
# -

# ### MSMAll_FIX
#

proc_type = 'MSMAll_FIX'
date_str = '2026-05-22' 
results_path = glue('/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}')

lasso_fix_df = read.csv(file.path(results_path, glue('{date_str}_lasso_fix_icc_df.csv'))) 
lasso_fix_icc <- icc_calc(lasso_fix_df)
lasso_fix_icc[[1]]$lambda
lasso_fix_icc[[2]]

pearson_fix_df = read.csv(file.path(results_path, glue('{date_str}_pearson_fix_icc_df.csv')) ) 
pearson_fix_icc <- icc_calc(pearson_fix_df) 
pearson_fix_icc[[1]]$lambda
pearson_fix_icc[[2]]

uoi_fix_df = read.csv(file.path(results_path, glue('{date_str}_uoi_fix_icc_df.csv'))) 
uoi_fix_icc <- icc_calc(uoi_fix_df) 
uoi_fix_icc[[1]]$lambda
uoi_fix_icc[[2]]

ridge_fix_df = read.csv(file.path(results_path, glue('{date_str}_ridge_fix_icc_df.csv')) ) 
ridge_fix_icc <- icc_calc(ridge_fix_df) 
ridge_fix_icc[[1]]$lambda
ridge_fix_icc[[2]]

pcorr_fix_df = read.csv(file.path(results_path, glue('{date_str}_pcorr_fix_icc_df.csv')) ) 
pcorr_fix_icc <- icc_calc(pcorr_fix_df) 
pcorr_fix_icc[[1]]$lambda
pcorr_fix_icc[[2]]

lassoThresh_fix_df = read.csv(file.path(results_path, glue('{date_str}_lassoThresh_fix_icc_df.csv')) ) 
lassoThresh_fix_icc <- icc_calc(lassoThresh_fix_df) 
lassoThresh_fix_icc[[1]]$lambda
lassoThresh_fix_icc[[2]]

uoiThresh_fix_df = read.csv(file.path(results_path, glue('{date_str}_uoiThresh_fix_icc_df.csv')) ) 
uoiThresh_fix_icc <- icc_calc(uoiThresh_fix_df) 
uoiThresh_fix_icc[[1]]$lambda
uoiThresh_fix_icc[[2]]

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### XCPD
# -

lasso_xcpd_df = read.csv(file.path(results_path, 'lasso_xcpd_icc_df.csv')) 
lasso_xcpd_icc <- icc_calc(lasso_xcpd_df)
lasso_xcpd_icc[[1]]$lambda
lasso_xcpd_icc[[2]]

pearson_xcpd_df = read.csv(file.path(results_path, 'pearson_xcpd_icc_df.csv')) 
pearson_xcpd_icc <- icc_calc(pearson_xcpd_df) 
pearson_xcpd_icc[[1]]$lambda
pearson_xcpd_icc[[2]]

uoi_xcpd_df = read.csv(file.path(results_path, 'uoi_xcpd_icc_df.csv')) 
uoi_xcpd_icc <- icc_calc(uoi_xcpd_df) 
uoi_xcpd_icc[[1]]$lambda
uoi_xcpd_icc[[2]]

ridge_xcpd_df = read.csv(file.path(results_path, 'ridge_xcpd_icc_df.csv')) 
ridge_xcpd_icc <- icc_calc(ridge_xcpd_df) 
ridge_xcpd_icc[[1]]$lambda
ridge_xcpd_icc[[2]]

# +
#make these into a dataframe and export for plotting 
# -

# ## Univariate ICC: 

assert 0 == 1

lasso_fix_results <- list()
for (ii in 1:10000) {
    lasso_fix_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=lasso_fix_df %>% filter(pos == ii) ) ) 
} 

lubridate::today()

lasso_fix_results[!sapply(lasso_fix_results, is.numeric)] = NaN
write.csv(lasso_fix_results,file=glue("results/{date}lasso_fix_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
lasso_xcpd_df = read.csv(file.path(results_path,
                                   'lasso_xcpd_icc_df.csv')) 
lasso_xcpd_results <- list()

for (ii in 1:10000) {
    lasso_xcpd_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=lasso_xcpd_df %>% filter(pos == ii) ) ) 
} 

lasso_xcpd_results[!sapply(lasso_xcpd_results, is.numeric)] = NaN
write.csv(lasso_xcpd_results,file="results/lasso_xcpd_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
pearson_xcpd_df = read.csv(file.path(results_path, 
                                     'pearson_xcpd_icc_df.csv')) 
results <- list()

for (ii in 1:10000) {

    results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=pearson_xcpd_df %>% filter(pos == ii) ) ) 
} 

results[!sapply(results, is.numeric)] = NaN
write.csv(results,file="results/pearson_xcpd_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
pearson_msmall_df = read.csv(file.path(results_path, 
                                       'pearson_fix_icc_df.csv')) 
pearson_msmall_results <- list()

for (ii in 1:10000) {

    pearson_msmall_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=pearson_msmall_df %>% filter(pos == ii) ) ) 
} 

pearson_msmall_results[!sapply(pearson_msmall_results, is.numeric)] = NaN
write.csv(pearson_msmall_results,file="results/pearson_fix_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
ridge_fix_df = read.csv(file.path(results_path, 'intermediate_results', 
                                       'ridge_fix_icc_df.csv')) 
ridge_fix_results <- list()

for (ii in 1:10000) {

    ridge_fix_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=ridge_fix_df %>% filter(pos == ii) ) ) 
} 

ridge_fix_results[!sapply(ridge_fix_results, is.numeric)] = NaN
write.csv(ridge_fix_results,file="results/ridge_fix_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
pcorr_fix_df = read.csv(file.path(results_path, 
                                       'pcorr_fix_icc_df.csv')) 
pcorr_fix_results <- list()

for (ii in 1:10000) {

    pcorr_fix_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=pcorr_fix_df %>% filter(pos == ii) ) ) 
} 

pcorr_fix_results[!sapply(pcorr_fix_results, is.numeric)] = NaN
write.csv(pcorr_fix_results,file="results/pcorr_fix_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
uoi_msmall_df = read.csv(file.path(results_path, 
                                   'uoi_fix_icc_df.csv')) 
uoi_msmall_results <- list()

for (ii in 1:10000) {

    uoi_msmall_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=uoi_msmall_df %>% filter(pos == ii) ) ) 
} 

uoi_msmall_results[!sapply(uoi_msmall_results, is.numeric)] = NaN
write.csv(uoi_msmall_results,file="results/uoi_fix_lme_icc.csv", row.names=FALSE, col.names=FALSE)

# +
uoi_xcpd_df = read.csv(file.path(results_path, 'uoi_xcpd_icc_df.csv')) 
uoi_xcpd_results <- list()

for (ii in 1:10000) {

    uoi_xcpd_results[[ii]] <- try(ICC2.lme(dv=values, grp=sub, data=uoi_msmall_df %>% filter(pos == ii) ) ) 
} 

uoi_xcpd_results[!sapply(uoi_xcpd_results, is.numeric)] = NaN
write.csv(uoi_xcpd_results,file="results/uoi_xcpd_lme_icc.csv", row.names=FALSE, col.names=FALSE)
# -










