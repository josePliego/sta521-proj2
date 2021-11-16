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

cv_logreg <- CVmaster(
  dt_train_logreg,
  mod_logreg,
  rec_logreg,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summary_logrec = cv_logreg %>%
  mutate(model = "Logistic Regression")

cv_logreg %>%
  summarise(across(.estimate, mean))

# Logistic Regression PCA

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

cv_lda <- CVmaster(
  dt_train_lda,
  mod_lda,
  rec_lda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summary_lda = cv_lda %>%
  mutate(model = "LDA")

cv_lda %>%
  summarise(across(.estimate, mean))

# PCA LDA

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


# 3. QDA ------------------------------------------------------------------

# QDA less blocks

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

cv_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summary_qda = cv_qda %>%
  mutate(model = "QDA")

cv_qda %>%
  summarise(across(.estimate, mean))

#QDA more blocks

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

cv_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "block",
  .columns = 10,
  .rows = 10
)

cv_qda

cv_qda %>%
  summarise(across(.estimate, mean))


#PCA QDA

rec_pca_qda <- rec_qda %>%
  step_pca(all_predictors(), num_comp = 8)

rec_pca_qda %>%
  prep() %>%
  juice() %>%
  pivot_longer(cols = -label) %>%
  ggplot(aes(x = value, fill = label)) +
  geom_density() +
  facet_wrap(~name, scales = "free")

cv_pca_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_pca_qda,
  .method = "kmeans",
  .k = 10
)

cv_pca_qda

cv_pca_qda %>%
  summarise(across(.estimate, mean))

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

cv_svm <- CVmaster(
  dt_train_svm,
  mod_svm,
  rec_svm,
  .method = "block",
  .columns = 2,
  .rows = 2
)

write_rds(cv_svm, "data/03_svm_cv.rds")


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
  learn_rate = tune(),
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

doParallel::registerDoParallel(cores = 4)

set.seed(42)
xgb_res <- tune_grid(
  xg_wf,
  resamples = cvsplit_xg_tune,
  grid = xg_grid
)

xgb_res$.notes

xg_grid <- xg_wf %>%
  parameters() %>%
  grid_regular()

xg_tune_fit <- xg_wf %>%
  tune_grid(
    resamples = cvsplit_tune,
    grid = xg_grid,
    metrics = metric_set(accuracy, roc_auc)
  )

write_rds(xg_tune_fit, "data/03_xg_tune.rds")


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

cv_nb <- CVmaster(
  dt_train_nb,
  mod_nb,
  rec_nb,
  .method = "block",
  .columns = 3,
  .rows = 3
)

summary_nb = cv_nb %>%
  mutate(model = "NaiveBayes")

cv_nb %>%
  summarise(across(.estimate, mean))

summary = cbind(summary_lda, summary_logrec, summary_nb, summary_qda)
