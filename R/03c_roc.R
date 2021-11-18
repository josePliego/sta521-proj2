## Script: 03b_modeling.R
## Inputs:
  # R/02c_CVmaster.R
  # cache/02_dt_train_block.rds
  # cache/02_dt_test_block.rds
  # cache/03_svm_tune.rds
  # cache/03_xg_best.rds
## Outputs:
  # cache/03_fit_xg.rds

# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(xgboost, pos = 998)
library(kernlab)
library(MASS, pos = 999)
library(klaR)
library(patchwork)

dt_train <- read_rds("cache/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~ if_else(img == "img3", .x + 299, .x))
  )

dt_test <- read_rds("cache/02_dt_test_block.rds") %>%
  select(-block)

set.seed(42)
# Logistic Regression -----------------------------------------------------

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

wf_logreg <- workflow() %>%
  add_model(mod_logreg) %>%
  add_recipe(rec_logreg)

fit_logreg <- wf_logreg %>%
  fit(dt_train_logreg)

roc_logreg <- fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_curve(label, .pred_0)

thres_logreg <- roc_logreg %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

logreg_auc <- fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_auc(label, .pred_0)

logreg_filter_auc <- roc_logreg %>%
  filter(abs(.threshold - thres_logreg) < 0.0001) %>%
  dplyr::slice(1)

logreg_roc <- roc_logreg %>%
  autoplot() +
  geom_point(
    data = logreg_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "Logistic Regression",
    subtitle = paste("AUC =", round(logreg_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_logreg.png",
  logreg_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_logreg <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_logreg %>%
  bind_cols(
    predict(fit_logreg, new_data = dt_test_logreg, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))


# LDA ---------------------------------------------------------------------

dt_train_lda <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_lda <- recipe(label ~ ., data = dt_train_lda) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_lda <- discrim_linear(
  mode = "classification",
  engine = "MASS"
)

wf_lda <- workflow() %>%
  add_model(mod_lda) %>%
  add_recipe(rec_lda)

fit_lda <- wf_lda %>%
  fit(dt_train_lda)

roc_lda <- fit_lda %>%
  augment(new_data = dt_train_lda) %>%
  roc_curve(label, .pred_0)

thres_lda <- roc_lda %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

lda_auc <- fit_lda %>%
  augment(new_data = dt_train_lda) %>%
  roc_auc(label, .pred_0)

lda_filter_auc <- roc_lda %>%
  filter(abs(.threshold - thres_lda) < 0.0001) %>%
  dplyr::slice(1)

lda_roc <- roc_lda %>%
  autoplot() +
  geom_point(
    data = lda_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "LDA",
    subtitle = paste("AUC =", round(lda_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_lda.png",
  lda_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_lda <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_lda %>%
  bind_cols(
    predict(fit_lda, new_data = dt_test_lda, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))


# QDA ---------------------------------------------------------------------

dt_train_qda <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_qda <- recipe(label ~ ., data = dt_train_qda) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_qda <- discrim_quad(
  mode = "classification",
  engine = "MASS"
)

wf_qda <- workflow() %>%
  add_model(mod_qda) %>%
  add_recipe(rec_qda)

fit_qda <- wf_qda %>%
  fit(dt_train_qda)

roc_qda <- fit_qda %>%
  augment(new_data = dt_train_qda) %>%
  roc_curve(label, .pred_0)

thres_qda <- roc_qda %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

qda_auc <- fit_qda %>%
  augment(new_data = dt_train_qda) %>%
  roc_auc(label, .pred_0)

qda_filter_auc <- roc_qda %>%
  filter(abs(.threshold - thres_qda) < 0.0001) %>%
  dplyr::slice(1)

qda_roc <- roc_qda %>%
  autoplot() +
  geom_point(
    data = qda_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "QDA",
    subtitle = paste("AUC =", round(qda_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_qda.png",
  qda_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_qda <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_qda %>%
  bind_cols(
    predict(fit_qda, new_data = dt_test_qda, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))


# SVM ---------------------------------------------------------------------

dt_train_svm <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_svm <- recipe(label ~ ., data = dt_train_svm) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

best_cost <- 0.177
best_sigma <- 1

mod_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = best_cost,
  rbf_sigma = best_sigma
)

wf_svm <- workflow() %>%
  add_model(mod_svm) %>%
  add_recipe(rec_svm)

fit_svm <- wf_svm %>%
  fit(dt_train_svm)

roc_svm <- fit_svm %>%
  augment(new_data = dt_train_svm) %>%
  roc_curve(label, .pred_0)

thres_svm <- roc_svm %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

svm_auc <- fit_svm %>%
  augment(new_data = dt_train_svm) %>%
  roc_auc(label, .pred_0)

svm_filter_auc <- roc_svm %>%
  filter(abs(.threshold - thres_svm) < 0.0001) %>%
  dplyr::slice(1)

svm_roc <- roc_svm %>%
  autoplot() +
  geom_point(
    data = svm_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "RBF SVM",
    subtitle = paste("AUC =", round(svm_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_svm.png",
  svm_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_svm <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_svm %>%
  bind_cols(
    predict(fit_svm, new_data = dt_test_svm, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))


# XGBoost ---------------------------------------------------------------------

dt_train_xgb <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_xgb <- recipe(label ~ ., data = dt_train_xgb) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

xg_tune_fit <- read_rds("cache/03_xg_tune.rds")
xg_tune_fit %>%
  select_best("accuracy")
xg_tune_best <- xg_tune_fit %>%
  select_best("roc_auc")

mod_xgb <- boost_tree(
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

wf_xgb <- workflow() %>%
  add_model(mod_xgb) %>%
  add_recipe(rec_xgb)

fit_xgb <- wf_xgb %>%
  fit(dt_train_xgb)

write_rds(fit_xgb, "cache/03_fit_xg.rds")

roc_xgb <- fit_xgb %>%
  augment(new_data = dt_train_xgb) %>%
  roc_curve(label, .pred_0)

thres_xgb <- roc_xgb %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

xgb_auc <- fit_xgb %>%
  augment(new_data = dt_train_xgb) %>%
  roc_auc(label, .pred_0)

xgb_filter_auc <- roc_xgb %>%
  filter(abs(.threshold - thres_xgb) < 0.0001) %>%
  dplyr::slice(1)

xgb_roc <- roc_xgb %>%
  autoplot() +
  geom_point(
    data = xgb_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "XGBoost",
    subtitle = paste("AUC =", round(xgb_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_xgb.png",
  xgb_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_xgb <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_xgb %>%
  bind_cols(
    predict(fit_xgb, new_data = dt_test_xgb, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))


# Naive Bayes -------------------------------------------------------------

dt_train_nb <- dt_train %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_nb <- recipe(label ~ ., data = dt_train_nb) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

mod_nb <- naive_Bayes(
  mode = "classification",
  engine = "klaR"
)

wf_nb <- workflow() %>%
  add_model(mod_nb) %>%
  add_recipe(rec_nb)

fit_nb <- wf_nb %>%
  fit(dt_train_nb)

roc_nb <- fit_nb %>%
  augment(new_data = dt_train_nb) %>%
  roc_curve(label, .pred_0)

thres_nb <- roc_nb %>%
  mutate(dist = sqrt((1 - specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

nb_auc <- fit_nb %>%
  augment(new_data = dt_train_nb) %>%
  roc_auc(label, .pred_0)

nb_filter_auc <- roc_nb %>%
  filter(abs(.threshold - thres_nb) < 0.0001) %>%
  dplyr::slice(1)

nb_roc <- roc_nb %>%
  autoplot() +
  geom_point(
    data = nb_filter_auc,
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    title = "Naive Bayes",
    subtitle = paste("AUC =", round(nb_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_nb.png",
  nb_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

dt_test_nb <- dt_test %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test_nb %>%
  bind_cols(
    predict(fit_nb, new_data = dt_test_nb, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_1 > 0.5, 1, 0))) %>%
  summarise(mean(label == pred))

roc_plots <- (logreg_roc|lda_roc|qda_roc)/(svm_roc|nb_roc|xgb_roc)

ggsave(
  "graphs/03_roc_plots.png",
  roc_plots,
  width = 2.99 * 10,
  height = 3.82 * 10,
  units = "cm"
)
