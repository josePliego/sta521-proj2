# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(xgboost, pos = 998)
library(kernlab)
library(MASS, pos = 999)
library(klaR)

dt_train <- read_rds("data/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~if_else(img == "img3", .x + 299, .x))
  )

# Logistic Regression -----------------------------------------------------

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

wf_logreg <- workflow() %>%
  add_model(mod_logreg) %>%
  add_recipe(rec_logreg)

fit_logreg <- wf_logreg %>%
  fit(dt_train_logreg)

roc_logreg <- fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_curve(label, .pred_0)

thres_logreg <- roc_logreg %>%
  mutate(dist = sqrt((1-specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

logreg_auc <- fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_auc(label, .pred_0)

logreg_filter_auc <- roc_logreg %>%
  filter(abs(.threshold - thres_logreg)<0.0001) %>%
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
    subtitle = paste("AUC =", round(logreg_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_logreg.png",
  logreg_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)



# LDA ---------------------------------------------------------------------

dt_train_lda <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_lda <- recipe(label ~., data = dt_train_lda) %>%
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
  mutate(dist = sqrt((1-specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

lda_auc <- fit_lda %>%
  augment(new_data = dt_train_lda) %>%
  roc_auc(label, .pred_0)

lda_filter_auc <- roc_lda %>%
  filter(abs(.threshold - thres_lda)<0.0001) %>%
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
    subtitle = paste("AUC =", round(lda_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_lda.png",
  lda_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)


# QDA ---------------------------------------------------------------------

dt_train_qda <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_qda <- recipe(label ~., data = dt_train_qda) %>%
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
  mutate(dist = sqrt((1-specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

qda_auc <- fit_qda %>%
  augment(new_data = dt_train_qda) %>%
  roc_auc(label, .pred_0)

qda_filter_auc <- roc_qda %>%
  filter(abs(.threshold - thres_qda)<0.0001) %>%
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
    subtitle = paste("AUC =", round(qda_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_qda.png",
  qda_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)


# XGBoost ---------------------------------------------------------------------

dt_train_xgb <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_xgb <- recipe(label ~., data = dt_train_xgb) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

xg_tune_fit <- read_rds("data/03_xg_tune.rds")
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

# Saved the trained model
# fit_xgb <- wf_xgb %>%
#   fit(dt_train_xgb)

fit_xgb <- read_rds("data/04_fit_xg.rds")

roc_xgb <- fit_xgb %>%
  augment(new_data = dt_train_xgb) %>%
  roc_curve(label, .pred_0)

thres_xgb <- roc_xgb %>%
  mutate(dist = sqrt((1-specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

xgb_auc <- fit_xgb %>%
  augment(new_data = dt_train_xgb) %>%
  roc_auc(label, .pred_0)

xgb_filter_auc <- roc_xgb %>%
  filter(abs(.threshold - thres_xgb)<0.0001) %>%
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
    subtitle = paste("AUC =", round(xgb_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_xgb.png",
  xgb_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)


# Naive Bayes -------------------------------------------------------------

dt_train_nb <- dt_train %>%
  mutate(across(label, ~if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

rec_nb <- recipe(label ~., data = dt_train_nb) %>%
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
  mutate(dist = sqrt((1-specificity)^2 + (sensitivity - 1)^2)) %>%
  arrange(dist) %>%
  dplyr::slice(1) %>%
  .$.threshold

nb_auc <- fit_nb %>%
  augment(new_data = dt_train_nb) %>%
  roc_auc(label, .pred_0)

nb_filter_auc <- roc_nb %>%
  filter(abs(.threshold - thres_nb)<0.0001) %>%
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
    subtitle = paste("AUC =", round(nb_auc$.estimate, 3))
  )

ggsave(
  "graphs/03_ROC_nb.png",
  nb_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)
