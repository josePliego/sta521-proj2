## Script: 03b_modeling.R
## Inputs:
  # R/02c_CVmaster.R
  # cache/02_dt_train_block.rds
  # cache/03_svm_tune.rds
  # cache/03_xg_best.rds
## Outputs:
  # cache/03_logreg_cvblock.rds
  # cache/03_logreg_cvkmeans.rds
  # cache/03_lda_cvblock.rds
  # cache/03_lda_cvkmeans.rds
  # cache/03_qda_cvblock.rds
  # cache/03_qda_cvkmeans.rds
  # cache/03_svm_cvblock.rds
  # cache/03_svm_cvkmeans.rds
  # cache/03_xg_cvblock.rds
  # cache/03_xg_cvkmeans.rds
  # cache/03_naiveb_cvblock.rds
  # cache/03_naiveb_cvkmeans.rds

# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(MASS, pos = 999)
library(klaR)
library(xtable)

source("R/02c_CVmaster.R")

dt_train <- read_rds("cache/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~ if_else(img == "img3", .x + 299, .x))
  )

dt_test <- read_rds("cache/02_dt_test_block.rds") %>%
  select(-block)

set.seed(42)

# 1. Logistic Regression --------------------------------------------------

dt_train_logreg <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_logreg <- recipe(label ~ ., data = dt_train_logreg) %>%
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

write_rds(summaryblock_logreg, "cache/03_logreg_cvblock.rds")

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

write_rds(summarykmeans_logreg, "cache/03_logreg_cvkmeans.rds")


# 2. LDA ------------------------------------------------------------------

dt_train_lda <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

mod_lda <- discrim_linear(
  mode = "classification",
  engine = "MASS"
)

rec_lda <- recipe(label ~ ., data = dt_train_lda) %>%
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

write_rds(summaryblock_lda, "cache/03_lda_cvblock.rds")

cvkmeans_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_lda,
  .method = "kmeans",
  .k = 9
)

summarykmeans_lda <- cvkmeans_lda %>%
  mutate(model = "LDA", method = "K-means")

write_rds(summarykmeans_lda, "cache/03_lda_cvkmeans.rds")


# 3. QDA ------------------------------------------------------------------

dt_train_qda <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_qda <- recipe(label ~ ., data = dt_train_qda) %>%
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

summaryblock_qda <- cvblock_qda %>%
  mutate(model = "QDA", method = "Block")

write_rds(summaryblock_qda, "cache/03_qda_cvblock.rds")

cvkmeans_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "kmeans",
  .k = 9
)

summarykmeans_qda <- cvkmeans_qda %>%
  mutate(model = "QDA", method = "K-means")

write_rds(summarykmeans_qda, "cache/03_qda_cvkmeans.rds")

# 4. SVM ------------------------------------------------------------------

dt_train_svm <- dt_train %>%
  mutate(across(label, factor))

rec_svm <- recipe(label ~ ., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

svm_tune_fit <- read_rds("cache/03_svm_tune.rds")

svm_wf <- workflow() %>%
  add_recipe(rec_svm) %>%
  add_model(mod_svm)

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

summaryblock_svm <- cvblock_svm %>%
  mutate(model = "SVM", method = "Block")

write_rds(cvblock_svm, "cache/03_svm_cvblock.rds")

cvkmeans_svm <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "kmeans",
  .k = 9
)

summarykmeans_svm <- cvkmeans_svm %>%
  mutate(model = "SVM", method = "K-means")

write_rds(cvkmeans_svm, "cache/03_svm_cvkmeans.rds")


# 5. XGBoost --------------------------------------------------------------

dt_train_xg <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_xg <- recipe(label ~ ., data = dt_train_xg) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

xg_tune_best <- read_rds("cache/03_xg_best.rds")

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

summaryblock_xg <- cvblock_xg %>%
  mutate(model = "xg", method = "Block")

write_rds(cvblock_xg, "cache/03_xg_cvblock.rds")

cvkmeans_xg <- CVmaster(
  dt_train_xg,
  mod_xg,
  rec_xg,
  .method = "kmeans",
  .k = 9
)

summarykmeans_xg <- cvkmeans_xg %>%
  mutate(model = "xg", method = "K-means")

write_rds(cvkmeans_xg, "cache/03_xg_cvkmeans.rds")


# 6. Naive Bayes ----------------------------------------------------------

dt_train_nb <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_nb <- recipe(label ~ ., data = dt_train_nb) %>%
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

summaryblock_nb <- cvblock_nb %>%
  mutate(model = "Naive Bayes", method = "Block")

write_rds(summaryblock_nb, "cache/03_naiveb_cvblock.rds")

cvkmeans_nb <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "kmeans",
  .k = 9
)

summarykmeans_nb <- cvkmeans_nb %>%
  mutate(model = "Naive Bayes", method = "K-means")

write_rds(summarykmeans_nb, "cache/03_naiveb_cvkmeans.rds")


# Summary Table -----------------------------------------------------------

summarykmeans_xgboost <- summarykmeans_xgboost %>%
  mutate(model = "XGBoost", method = "K-means")

summaryblock_xgboost <- summaryblock_xgboost %>%
  mutate(model = "XGBoost", method = "Block")

summary_kmeans <- cbind(
  summarykmeans_logreg[, 3], summarykmeans_lda[, 3], summarykmeans_qda[, 3],
  summarykmeans_nb[, 3], summarykmeans_svm[, 3], summarykmeans_xgboost[, 3]
)
colnames(summary_kmeans) <- c(
  "Logistic Regression", "LDA", "QDA", "Naive Bayes", "SVM", "XGBoost"
  )

summary_kmeans_averages <- rbind(
  summary_kmeans,
  summary_kmeans %>%
    apply(2, mean, na.rm = TRUE)
)

summary_block <- cbind(
  summaryblock_logreg[, 3], summaryblock_lda[, 3], summaryblock_qda[, 3],
  summaryblock_nb[, 3], summaryblock_svm[, 3], summaryblock_xgboost[, 3]
)
colnames(summary_block) <- c(
  "Logistic Regression", "LDA", "QDA", "Naive Bayes", "SVM", "XGBoost"
  )

summary_block_averages <- rbind(
  summary_block,
  summary_block %>%
    apply(2, mean, na.rm = TRUE)
)

summary_block_averages %>% xtable::xtable()

summary_kmeans_averages %>% xtable::xtable()
