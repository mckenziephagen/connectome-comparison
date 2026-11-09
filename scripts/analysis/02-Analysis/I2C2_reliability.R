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

# %% vscode={"languageId": "r"}
renv::load()

# %% vscode={"languageId": "r"}
library('I2C2') 
library('tidyverse') 
library('glue')
library('ini')
config <- read.ini("../../../config.ini")


# %% vscode={"languageId": "r"}
i2c2_calc <- function(df) { 

    df$pos <- NULL
    df$movement <- NULL

    wide_df <- pivot_wider(data=df, names_from='X', values_from='values')

    y <- select(wide_df, -sub, -ses)
    id <- wide_df$sub
    visit <- wide_df$ses
    
    i2c2_res <- I2C2(y, id = id, visit = visit, demean = TRUE) 
    i2c2_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI #na.rm = TRUE if err
    
    icc_obj = list(i2c2_res, i2c2_ci)

return(icc_obj) 
 } 

# %% [markdown]
# ### MSMAll_fix
#

# %% vscode={"languageId": "r"}
results_path  <- file.path(config$Paths$script_dir, 'results')
plotting_path <- file.path(config$Paths$script_dir, 'results', 'plotting')

# %% vscode={"languageId": "r"}
lasso_fix_df <- read.csv(file.path(results_path, 'MSMAll-fix', 'proc-fix_model-lassoBIC_reliability-edges.csv')) 
pearson_fix_df <- read.csv(file.path(results_path, 'MSMAll-fix', 'proc-fix_model-Pearson_reliability-edges.csv')) 
uoi_fix_df <- read.csv(file.path(results_path, 'MSMAll-fix', 'proc-fix_model-uoiLasso_reliability-edges.csv')) 
ridge_fix_df <- read.csv(file.path(results_path, 'MSMAll-fix', 'proc-fix_model-ridgeCV_reliability-edges.csv')) 
pcorr_fix_df <- read.csv(file.path(results_path, 'MSMAll-fix', 'proc-fix_model-partial-correlation_reliability-edges.csv')) 

# %% [markdown]
# #### I2C2

# %% vscode={"languageId": "r"}
lasso_fix_i2c2 <- i2c2_calc(lasso_fix_df)
lasso_fix_i2c2[[1]]$lambda
lasso_fix_i2c2[[2]]

# %% vscode={"languageId": "r"}
pearson_fix_i2c2 <- i2c2_calc(pearson_fix_df) 
pearson_fix_i2c2[[1]]$lambda
pearson_fix_i2c2[[2]]

# %% vscode={"languageId": "r"}
uoi_fix_i2c2 <- i2c2_calc(uoi_fix_df) 
uoi_fix_i2c2[[1]]$lambda
uoi_fix_i2c2[[2]]

# %% vscode={"languageId": "r"}
ridge_fix_i2c2 <- i2c2_calc(ridge_fix_df) 
ridge_fix_i2c2[[1]]$lambda
ridge_fix_i2c2[[2]]

# %% vscode={"languageId": "r"}
pcorr_fix_i2c2 <- i2c2_calc(pcorr_fix_df) 
pcorr_fix_i2c2[[1]]$lambda
pcorr_fix_i2c2[[2]]

# %% vscode={"languageId": "r"}
i2c2_df = data.frame(LASSO = c(lasso_fix_i2c2[[1]][['lambda']], lasso_fix_i2c2[[2]]), 
             Ridge = c(ridge_fix_i2c2[[1]][['lambda']], ridge_fix_i2c2[[2]]), 
            UoI = c(uoi_fix_i2c2[[1]][['lambda']], uoi_fix_i2c2[[2]]), 
            Pcorr = c(pcorr_fix_i2c2[[1]][['lambda']], pcorr_fix_i2c2[[2]]), 
            Pearson = c(pearson_fix_i2c2[[1]][['lambda']], pearson_fix_i2c2[[2]]), 

             row.names = c('lambda', '2.5', '97.5')) 

write.csv(i2c2_df, file.path(results_path, 'i2c2_fix_df.csv'))

# %% [markdown]
# ### XCPD

# %% vscode={"languageId": "r"}
proc_type = 'xcpd'

# %% vscode={"languageId": "r"}
lasso_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-lassoBIC_reliability-edges.csv')) 
pearson_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-Pearson_reliability-edges.csv')) 
uoi_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-uoiLasso_reliability-edges.csv')) 
ridge_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-ridgeCV_reliability-edges.csv')) 
pcorr_xcpd_df <- read.csv(file.path(results_path, 'XCPD', 'proc-XCPD_model-partial-correlation_reliability-edges.csv')) 


# %% vscode={"languageId": "r"}
lasso_xcpd_i2c2 <- i2c2_calc(lasso_xcpd_df)
lasso_xcpd_i2c2[[1]]$lambda
lasso_xcpd_i2c2[[2]]

# %% vscode={"languageId": "r"}
pearson_xcpd_i2c2 <- i2c2_calc(pearson_xcpd_df) 
pearson_xcpd_i2c2[[1]]$lambda
pearson_xcpd_i2c2[[2]]

# %% vscode={"languageId": "r"}
uoi_xcpd_i2c2 <- i2c2_calc(uoi_xcpd_df) 
uoi_xcpd_i2c2[[1]]$lambda
uoi_xcpd_i2c2[[2]]

# %% vscode={"languageId": "r"}
ridge_xcpd_i2c2 <- i2c2_calc(ridge_xcpd_df) 
ridge_xcpd_i2c2[[1]]$lambda
ridge_xcpd_i2c2[[2]]

# %% vscode={"languageId": "r"}
pcorr_xcpd_i2c2 <- i2c2_calc(pcorr_xcpd_df) 
pcorr_xcpd_i2c2[[1]]$lambda
pcorr_xcpd_i2c2[[2]]

# %% vscode={"languageId": "r"}
i2c2_df = data.frame(LASSO = c(lasso_xcpd_i2c2[[1]][['lambda']], lasso_xcpd_i2c2[[2]]), 
             Ridge = c(ridge_xcpd_i2c2[[1]][['lambda']], ridge_xcpd_i2c2[[2]]), 
            UoI = c(uoi_xcpd_i2c2[[1]][['lambda']], uoi_xcpd_i2c2[[2]]), 
            Pcorr = c(pcorr_xcpd_i2c2[[1]][['lambda']], pcorr_xcpd_i2c2[[2]]), 
            Pearson = c(pearson_xcpd_i2c2[[1]][['lambda']], pearson_xcpd_i2c2[[2]]), 

             row.names = c('lambda', '2.5', '97.5')) 

write.csv(i2c2_df, file.path(results_path, 'i2c2_xcpd_df.csv'))

# %% [markdown]
# ### PNC

# %% vscode={"languageId": "r"}
lasso_pnc_df <- read.csv(file.path(results_path, 'PNC_XCPD', 'proc-pnc_model-lassoBIC_reliability-edges.csv')) 
pearson_pnc_df <- read.csv(file.path(results_path, 'PNC_XCPD', 'proc-pnc_model-Pearson_reliability-edges.csv')) 
uoi_pnc_df <- read.csv(file.path(results_path, 'PNC_XCPD', 'proc-pnc_model-uoiLasso_reliability-edges.csv')) 
ridge_pnc_df <- read.csv(file.path(results_path, 'PNC_XCPD', 'proc-pnc_model-ridgeCV_reliability-edges.csv')) 
pcorr_pnc_df <- read.csv(file.path(results_path, 'PNC_XCPD', 'proc-pnc_model-partial-correlation_reliability-edges.csv')) 

# %% vscode={"languageId": "r"}

# %% vscode={"languageId": "r"}
lasso_pnc_i2c2 <- i2c2_calc(na.omit(lasso_pnc_df))
lasso_pnc_i2c2[[1]]$lambda
lasso_pnc_i2c2[[2]]

# %% vscode={"languageId": "r"}
pearson_pnc_i2c2 <- i2c2_calc(na.omit(pearson_pnc_df))
pearson_pnc_i2c2[[1]]$lambda
pearson_pnc_i2c2[[2]]

# %% vscode={"languageId": "r"}
ridge_pnc_i2c2 <- i2c2_calc(na.omit(ridge_pnc_df))
ridge_pnc_i2c2[[1]]$lambda
ridge_pnc_i2c2[[2]]

# %% vscode={"languageId": "r"}
uoi_pnc_i2c2 <- i2c2_calc(na.omit(uoi_pnc_df))
uoi_pnc_i2c2[[1]]$lambda
uoi_pnc_i2c2[[2]]

# %% vscode={"languageId": "r"}
pcorr_pnc_i2c2 <- i2c2_calc(na.omit(pcorr_pnc_df))
pcorr_pnc_i2c2[[1]]$lambda
pcorr_pnc_i2c2[[2]]

# %% vscode={"languageId": "r"}
i2c2_df = data.frame(LASSO = c(lasso_pnc_i2c2[[1]][['lambda']], lasso_pnc_i2c2[[2]]), 
             Ridge = c(ridge_pnc_i2c2[[1]][['lambda']], ridge_pnc_i2c2[[2]]), 
            UoI = c(uoi_pnc_i2c2[[1]][['lambda']], uoi_pnc_i2c2[[2]]), 
            Pcorr = c(pcorr_pnc_i2c2[[1]][['lambda']], pcorr_pnc_i2c2[[2]]), 
            Pearson = c(pearson_pnc_i2c2[[1]][['lambda']], pearson_pnc_i2c2[[2]]), 

             row.names = c('lambda', '2.5', '97.5')) 

write.csv(i2c2_df, file.path(results_path, 'i2c2_pnc_df.csv')) 
