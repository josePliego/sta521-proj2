# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(MASS, pos = 999)
library(klaR)

source("R/02_CVmaster.R")

dt_train <- read_rds("data/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~if_else(img == "img3", .x + 299, .x))
  )

dt_test <- read_rds("data/02_dt_test_block.rds") %>%
  select(-block)

set.seed(42)

# 1. Logistic Regression --------------------------------------------------

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

cvblock_logreg <- CVmaster(
  dt_train_logreg,
  mod_logreg,
  rec_logreg,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_logreg <- cvblock_logreg %>%
  mutate(
    model = "Logistic Regression",
    method = "Block"
    )

write_rds(summaryblock_logreg, "data/03_logreg_cvblock.rds")

cvkmeans_logreg <- CVmaster(
  dt_train_logreg,
  mod_logreg,
  rec_logreg,
  .method = "kmeans",
  .k = 9
)

summarykmeans_logreg <- cvkmeans_logreg %>%
  mutate(
    model = "Logistic Regression",
    method = "k-means"
  )

write_rds(summarykmeans_logreg, "data/03_logreg_cvkmeans.rds")


# 2. LDA ------------------------------------------------------------------

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

cvblock_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_lda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_lda <- cvblock_lda %>%
  mutate(model = "LDA", method = "Block")

write_rds(summaryblock_lda, "data/03_lda_cvblock.rds")

cvkmeans_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_lda,
  .method = "kmeans",
  .k = 9
)

summarykmeans_lda <- cvkmeans_lda %>%
  mutate(model = "LDA", method = "K-means")

write_rds(summarykmeans_lda, "data/03_lda_cvkmeans.rds")


# 3. QDA ------------------------------------------------------------------

dt_train_qda <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_qda <- recipe(label ~., data = dt_train_qda) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_qda <- discrim_quad(
  mode = "classification",
  engine = "MASS",
  regularization_method = NULL
)

cvblock_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_qda = cvblock_qda %>%
  mutate(model = "QDA", method = "Block")

write_rds(summaryblock_qda, "data/03_qda_cvblock.rds")

cvkmeans_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "kmeans",
  .k = 9
)

summarykmeans_qda = cvkmeans_qda %>%
  mutate(model = "QDA", method = "K-means")

write_rds(summarykmeans_qda, "data/03_qda_cvkmeans.rds")

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

write_rds(svm_tune_fit, "data/03_svm_tune.rds")

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

svm_tune_fit$.notes

svm_tune_fit %>%
  select(id, .metrics) %>%
  unnest(.metrics) %>%
  filter(.metric == "accuracy") %>%
  arrange(-.estimate)

svm_tune_fit %>%
  select(id, .metrics) %>%
  unnest(.metrics) %>%
  filter(.metric == "roc_auc") %>%
  arrange(-.estimate)

best_cost <- 0.177
best_sigma <- 1

mod_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = best_cost,
  rbf_sigma = best_sigma
)

cvblock_svm <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_svm = cvblock_svm %>%
  mutate(model = "SVM", method = "Block")

write_rds(cvblock_svm, "data/03_svm_cvblock.rds")

cvkmeans_svm <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "kmeans",
  .k = 9
)

summarykmeans_svm = cvkmeans_svm %>%
  mutate(model = "SVM", method = "K-means")


write_rds(cvkmeans_svm, "data/03_svm_cvkmeans.rds")


# 5. XGBoost --------------------------------------------------------------

dt_train_xg <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
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

rec_xg <- recipe(label ~., data = dt_train_xg) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

# Tuning
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

write_rds(xgb_res, "data/03_xg_tune.rds")
xg_tune_fit <- read_rds("data/03_xg_tune.rds")
xg_tune_fit %>%
  select_best("accuracy")
xg_tune_best <- xg_tune_fit %>%
  select_best("roc_auc")
write_rds(xg_tune_best, "data/03_xg_best.rds")
# Same best model for accuracy and roc

# Final fit
mod_xg <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  trees = 1000,
  tree_depth = xg_tune_best$tree_depth,
  min_n = xg_tune_best$min_n,
  loss_reduction = xg_tune_best$loss_reduction,
  sample_size = xg_tune_best$sample_size,
  mtry = xg_tune_best$mtry,
  learn_rate = xg_tune_best$learn_rate
)

cvblock_xg <- CVmaster(
  dt_train_xg,
  mod_xg,
  rec_xg,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_xg = cvblock_xg %>%
  mutate(model = "xg", method = "Block")

write_rds(cvblock_xg, "data/03_xg_cvblock.rds")

cvkmeans_xg <- CVmaster(
  dt_train_xg,
  mod_xg,
  rec_xg,
  .method = "kmeans",
  .k = 9
)

summarykmeans_xg = cvkmeans_xg %>%
  mutate(model = "xg", method = "K-means")


write_rds(cvkmeans_xg, "data/03_xg_cvkmeans.rds")


# 6. Naive Bayes ----------------------------------------------------------

dt_train_nb <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_nb <- recipe(label ~., data = dt_train_nb) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_nb <- naive_Bayes(
  mode = "classification",
  engine = "klaR",
  smoothness = NULL,
  Laplace = NULL
)

cvblock_nb <- CVmaster(
  dt_train_nb,
  mod_nb,
  rec_nb,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summaryblock_nb = cvblock_nb %>%
  mutate(model = "Naive Bayes", method = "Block")

write_rds(summaryblock_nb, "data/03_naiveb_cvblock.rds")

cvkmeans_nb <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "kmeans",
  .k = 9
)

summarykmeans_nb = cvkmeans_nb %>%
  mutate(model = "Naive Bayes", method = "K-means")

write_rds(summarykmeans_nb, "data/03_naiveb_cvkmeans.rds")

summarykmeans_xgboost = readRDS("data/03_xg_cvkmeans.rds")

summarykmeans_xgboost = summarykmeans_xgboost %>%
  mutate(model = "XGBoost", method = "K-means")

summaryblock_xgboost = readRDS("data/03_xg_cvblock.rds")

summaryblock_xgboost = summaryblock_xgboost %>%
  mutate(model = "XGBoost", method = "Block")



