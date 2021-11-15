# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(MASS, pos = 999)

source("R/02_CVmaster.R")

dt_train <- read_rds("data/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~if_else(img == "img3", .x + 299, .x))
  )

# Logistic Regression
dt_train_logreg <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_logreg <- recipe(label ~., data = dt_train_logreg) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_logreg <- logistic_reg(
  mode = "classification",
  engine = "glm"
  )

cv_logreg <- CVmaster(
  dt_train_logreg,
  mod_logreg,
  rec_logreg,
  .method = "block",
  .columns = 3,
  .rows = 3
)

cv_logreg

cv_logreg %>%
  summarise(across(.estimate, mean))

rec_pca_logreg <- rec_logreg %>%
  step_pca(all_predictors(), num_comp = 6)

cv_pca_logreg <- CVmaster(
  dt_train_logreg,
  mod_logreg,
  rec_pca_logreg,
  .method = "block",
  .columns = 3,
  .rows = 3
)

cv_pca_logreg

cv_pca_logreg %>%
  summarise(across(.estimate, mean))

# LDA

dt_train_lda <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

mod_lda <- discrim_linear(
  mode = "classification",
  engine = "MASS"
)

rec_lda <- recipe(label ~., data = dt_train_lda) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

cv_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_lda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

cv_lda

cv_lda %>%
  summarise(across(.estimate, mean))

rec_pca_lda <- rec_lda %>%
  step_pca(all_predictors(), num_comp = 8)

rec_pca_lda %>%
  prep() %>%
  juice() %>%
  pivot_longer(cols = -label) %>%
  ggplot(aes(x = value, fill = label)) +
  geom_density() +
  facet_wrap(~name, scales = "free")

cv_pca_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_pca_lda,
  .method = "kmeans",
  .k = 10
)

cv_pca_lda

cv_pca_lda %>%
  summarise(across(.estimate, mean))

# QDA

# 4. SVM ------------------------------------------------------------------
set.seed(42)
dt_train_svm <- dt_train %>%
  mutate(across(label, factor))

cvsplit_svm <- make_cvsplits(dt_train_svm, .method = "kmeans", .k = 5)

dt_subsample <- assessment(cvsplit_svm[1, 1]$splits[[1]])

rec_svm <- recipe(label ~., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

# Tuning

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

# cv_svm <- CVmaster(
#   dt_train_svm,
#   mod_svm,
#   rec_svm,
#   .method = "block",
#   .columns = 2,
#   .rows = 1
# )

svm_wf <- workflow() %>%
  add_recipe(rec_svm) %>%
  add_model(mod_svm)

start_grid <-
  svm_wf %>%
  parameters() %>%
  update(
    cost = cost(c(-6, 1))
    ) %>%
  grid_regular(levels = 3)

svm_wflow <- svm_wf %>%
  tune_grid(
    resamples = make_splits(dt_train_svm, .method = "kmeans", .k = 2),
    grid = start_grid,
    metrics = metric_set(accuracy)
    )


workflow() %>%
  add_recipe(rec_svm) %>%
  add_model(mod_svm) %>%
  fit(dt_train_svm)

cv_lda

cv_lda %>%
  summarise(across(.estimate, mean))

# Random Forest
# Boosted Trees
