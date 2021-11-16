devtools::install_github("tidymodels/parsnip")
devtools::install_github("tidymodels/discrim")
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

cv_qda

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



# SVM

dt_train_svm <- dt_train %>%
  mutate(across(label, factor))

mod_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = 1
)

rec_svm <- recipe(label ~., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

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

dt_train_rf <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

splitss = make_cvsplits(dt_train_rf, "block", 5, 1)

rec_rf <- recipe(label ~., data = dt_train_rf) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_rf <- discrim_quad(
  mode = "classification",
  engine = "ranger",

)



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

cv_qda

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



# SVM

dt_train_svm <- dt_train %>%
  mutate(across(label, factor))

mod_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = 1
)

rec_svm <- recipe(label ~., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

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

dt_train_rf <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

splitss = make_cvsplits(dt_train_rf, "block", 5, 1)

rec_rf <- recipe(label ~., data = dt_train_rf) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_rf <- discrim_quad(
  mode = "classification",
  engine = "ranger",

)

cv_qda <- CVmaster(
  dt_train_qda,
  mod_qda,
  rec_qda,
  .method = "block",
  .columns = 3,
  .rows = 3
)

# Naive Bayes

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

cv_nb

cv_nb %>%
  summarise(across(.estimate, mean))
