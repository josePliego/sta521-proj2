## Script: 03a_model_tuning.R
## Inputs:
# R/02c_CVmaster.R
# cache/02_dt_train_block.rds
## Outputs:
# cache/03_svm_tune.rds
# cache/03_xg_tune.rds
# cache/03_xg_best.rds

library(tidymodels)
library(tidyverse)

source("R/02c_CVmaster.R")

dt_train <- read_rds("cache/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~ if_else(img == "img3", .x + 299, .x))
  )

# 1. SVM ------------------------------------------------------------------

set.seed(42)
dt_train_svm <- dt_train %>%
  mutate(across(label, factor))

cvsplit_svm <- make_cvsplits(dt_train_svm, .method = "kmeans", .k = 5)

dt_subsample <- assessment(cvsplit_svm[1, 1]$splits[[1]])

rec_svm <- recipe(label ~ ., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

cvsplit_tune <- make_cvsplits(dt_subsample, .method = "kmeans", .k = 5)
mod_svm_tune <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)

svm_wf <- workflow() %>%
  add_recipe(rec_svm) %>%
  add_model(mod_svm_tune)

svm_grid <- svm_wf %>%
  parameters() %>%
  grid_regular()

svm_tune_fit <- svm_wf %>%
  tune_grid(
    resamples = cvsplit_tune,
    grid = svm_grid,
    metrics = metric_set(accuracy, roc_auc)
  )

write_rds(svm_tune_fit, "cache/03_svm_tune.rds")


# 2. XGBoost --------------------------------------------------------------

dt_train_xg <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

set.seed(43)
cvsplit_xg <- make_cvsplits(
  dt_train_xg,
  .method = "block",
  .columns = 2,
  .rows = 2
)
rand_index_xg <- sample(1:NROW(cvsplit_xg), size = 1)

subsample_xg <- assessment(
  cvsplit_xg[rand_index_xg, 1]$splits[[1]]
)

rec_xg <- recipe(label ~ ., data = dt_train_xg) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

cvsplit_xg_tune <- make_cvsplits(
  subsample_xg,
  .method = "block",
  .columns = 4,
  .rows = 4
)

mod_xg_tune <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
)

xg_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), dt_train),
  learn_rate(),
  size = 30
)

xg_wf <- workflow() %>%
  add_recipe(rec_xg) %>%
  add_model(mod_xg_tune)

doParallel::registerDoParallel(cores = 2)

set.seed(42)
xgb_res <- tune_grid(
  xg_wf,
  resamples = cvsplit_xg_tune,
  grid = xg_grid,
  metrics = metric_set(accuracy, roc_auc)
)

write_rds(xgb_res, "cache/03_xg_tune.rds")

xg_tune_fit %>%
  select_best("accuracy")
xg_tune_best <- xg_tune_fit %>%
  select_best("roc_auc")

write_rds(xg_tune_best, "cache/03_xg_best.rds")
