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

library('I2C2') 
library('tidyverse') 


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

# ### MSMAll

# +
results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'

lasso_msmall_df = read.csv(file.path(results_path, 'lasso_msmall_icc_df.csv')) 
lasso_msmall_icc <- icc_calc(lasso_msmall_df)
lasso_msmall_icc[[1]]$lambda
lasso_msmall_icc[[2]]
# -

pearson_msmall_df = read.csv(file.path(results_path, 'pearson_msmall_icc_df.csv')) 
pearson_msmall_icc <- icc_calc(pearson_msmall_df) 
pearson_msmall_icc[[1]]$lambda
pearson_msmall_icc[[2]]

uoi_msmall_df = read.csv(file.path(results_path, 'uoi_msmall_icc_df.csv')) 
uoi_msmall_icc <- icc_calc(uoi_msmall_df) 
uoi_msmall_icc[[1]]$lambda
uoi_msmall_icc[[2]]

# #### XCPD

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

# +
#make these into a dataframe and export for plotting 
# -


