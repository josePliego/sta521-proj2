## Script: 03b_modeling.R
## Inputs:
  # cache/01_dt_full.rds
  # R/02c_CVmaster.R
  # cache/02_dt_train_block.rds
  # cache/02_dt_test_block.rds
  # cache/03_xg_best.rds
  # cache/03_fit_xg.rds
## Outputs:

library(tidymodels)
library(tidyverse)
library(xgboost)

source("R/02c_CVmaster.R")

set.seed(42)

dt_train <- read_rds("cache/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~ if_else(img == "img3", .x + 299, .x))
  ) %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_test <- read_rds("cache/02_dt_test_block.rds") %>%
  select(-block) %>%
  mutate(across(label, ~ if_else(.x == -1, 0, 1))) %>%
  mutate(across(label, factor))

dt_full <- read_rds("cache/01_dt_full.rds")

rec_xg <- recipe(label ~ ., data = dt_train) %>%
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

wf_xg <- workflow() %>%
  add_recipe(rec_xg) %>%
  add_model(mod_xg)

fit_xg <- read_rds("cache/03_fit_xg.rds")

dt_test_bind <- dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  )

test_loss <- dt_test_bind %>%
  mn_log_loss(label, .pred_0) %>%
  .$.estimate

fit_obj <- extract_fit_parsnip(fit_xg)$fit
xgb_loss <- fit_obj$evaluation_log %>%
  as_tibble() %>%
  ggplot() +
  geom_line(aes(x = iter, y = training_logloss)) +
  geom_hline(yintercept = test_loss, linetype = 2) +
  labs(x = "Iteration", y = "Loss") +
  theme_bw()

ggsave(
  "graphs/04_xg_loss.png",
  xgb_loss,
  width = 2.99 * 5,
  height = 3.82 * 2,
  units = "cm"
)

xgb_importance <-
  xgb.importance(feature_names = fit_obj$feature_names, model = fit_obj) %>%
  as_tibble() %>%
  ggplot(aes(x = Gain, y = fct_reorder(Feature, Gain))) +
  geom_col(aes(fill = "a")) +
  labs(y = "Feature") +
  scale_fill_viridis_d() +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  "graphs/04_xg_importance.png",
  xgb_importance,
  width = 2.99 * 5,
  height = 3.82 * 2,
  units = "cm"
)

plot_deepness <- xgb.ggplot.deepness(fit_obj, which = "2x1")

leafs_plot <- plot_deepness[[1]] +
  aes(fill = "a") +
  scale_fill_viridis_d() +
  scale_y_continuous(labels = label_comma()) +
  scale_x_continuous(breaks = 1:10) +
  labs(x = "Leaf depth", title = "") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  "graphs/04_xg_leafdist.png",
  leafs_plot,
  width = 2.99 * 5,
  height = 3.82 * 2,
  units = "cm"
)

obs_plot <- plot_deepness[[2]] +
  aes(fill = "a") +
  scale_fill_viridis_d() +
  scale_x_continuous(breaks = 1:10) +
  labs(x = "Leaf depth", title = "") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  "graphs/04_xg_avgobs.png",
  obs_plot,
  width = 2.99 * 5,
  height = 3.82 * 2,
  units = "cm"
)

dt_aug <- fit_xg %>%
  augment(dt_train)

pca <- dt_aug %>%
  select(ndai:rad_an) %>%
  prcomp(scale. = TRUE)

pca_tib <- pca$x %>%
  as_tibble() %>%
  mutate(pred = dt_aug$.pred_class)

pca_tib %>%
  ggplot(aes(x = PC1, y = PC2, color = pred)) +
  geom_point()

make.grid <- function(x, n = 200) {
  grange <- apply(x, 2, range)
  x1 <- seq(from = grange[1, 1], to = grange[2, 1], length = n)
  x2 <- seq(from = grange[1, 2], to = grange[2, 2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

grid <- make.grid(pca$x)
proj_features <- cbind(
  grid,
  rnorm(n = NROW(grid), mean = mean(pca$x[, 3]), sd = sd(pca$x[, 3])),
  rnorm(n = NROW(grid), mean = mean(pca$x[, 4]), sd = sd(pca$x[, 4])),
  rnorm(n = NROW(grid), mean = mean(pca$x[, 5]), sd = sd(pca$x[, 5])),
  rnorm(n = NROW(grid), mean = mean(pca$x[, 6]), sd = sd(pca$x[, 6])),
  rnorm(n = NROW(grid), mean = mean(pca$x[, 7]), sd = sd(pca$x[, 7])),
  rnorm(n = NROW(grid), mean = mean(pca$x[, 8]), sd = sd(pca$x[, 8]))
)

proj_matrix <- as.matrix(proj_features) %*% t(pca$rotation)
proj_matrix <- proj_matrix %>%
  as_tibble() %>%
  mutate(x = 0, y = 0, img = "1")
pred_class_proj <- proj_matrix %>%
  bind_cols(predict(fit_xg, new_data = proj_matrix, type = "class"))

decision_boundary <- grid %>%
  mutate(pred = pred_class_proj$.pred_class) %>%
  ggplot() +
  geom_point(aes(x = X1, y = X2, color = pred), shape = 3) +
  geom_point(
    data = pca_tib,
    aes(x = PC1, y = PC2, color = pred),
    alpha = 0.3
  ) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_d() +
  labs(x = "PC1", y = "PC2", color = "Prediction") +
  theme_bw()

ggsave(
  "graphs/04_decision_boundary.png",
  decision_boundary,
  width = 20,
  height = 15,
  units = "cm"
)

test_auc <- dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  ) %>%
  roc_auc(label, .pred_0)

test_roc <- dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  ) %>%
  roc_curve(label, .pred_0)

xg_roc <- test_roc %>%
  autoplot() +
  geom_point(
    data = test_roc %>% filter(abs(.threshold - 0.5) < 0.001) %>% dplyr::slice(1),
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  ) +
  labs(
    subtitle = paste("AUC =", round(test_auc$.estimate, 3))
  )

ggsave(
  "graphs/04_xg_roc.png",
  xg_roc,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

test_probs <- dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  ) %>%
  ggplot(aes(x = x, y = y)) +
  geom_point(aes(color = .pred_1), size = 0.6) +
  labs(x = "", y = "", color = "Cloud Probability") +
  scale_color_viridis_c() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "top")

ggsave(
  "graphs/04_test_probs.png",
  test_probs,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_0 > 0.5, 0, 1))) %>%
  summarise(mean(pred == label))

dt_full_preds <- dt_full %>%
  bind_cols(
    predict(fit_xg, new_data = dt_full, type = "prob")
  )

probimg1 <- dt_full_preds %>%
  filter(img == "img1") %>%
  mutate(
    prob_cloud = case_when(
      # label == -1 ~ 0,
      # label == 1 ~ 1,
      TRUE ~ .pred_1
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = prob_cloud)) +
  geom_point() +
  labs(x = "", y = "", color = "Cloud Probability") +
  scale_color_viridis_c() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "top")

ggsave(
  "graphs/04_probimg1.png",
  probimg1,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

probimg2 <- dt_full_preds %>%
  filter(img == "img2") %>%
  mutate(
    prob_cloud = case_when(
      # label == -1 ~ 0,
      # label == 1 ~ 1,
      TRUE ~ .pred_1
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = prob_cloud)) +
  geom_point() +
  labs(x = "", y = "", color = "Cloud Probability") +
  scale_color_viridis_c() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "top")

ggsave(
  "graphs/04_probimg2.png",
  probimg2,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

probimg3 <- dt_full_preds %>%
  filter(img == "img3") %>%
  mutate(
    prob_cloud = case_when(
      # label == -1 ~ 0,
      # label == 1 ~ 1,
      TRUE ~ .pred_1
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = prob_cloud)) +
  geom_point() +
  labs(x = "", y = "", color = "Cloud Probability") +
  scale_color_viridis_c() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "top")

ggsave(
  "graphs/04_probimg3.png",
  probimg3,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

dt_test_misclas <- dt_test %>%
  bind_cols(
    predict(fit_xg, new_data = dt_test, type = "prob")
  ) %>%
  mutate(pred = factor(if_else(.pred_0 > 0.5, 0, 1)))

my_scale <- function(x) {
  (x - mean(x)) / sd(x)
}

table(dt_test_misclas$label, dt_test_misclas$pred)

dt_test_misclas %>%
  mutate(across(ndai, my_scale)) %>%
  ggplot(aes(x = ndai, fill = false_negative)) +
  geom_density(alpha = 0.5)

dt_test_misclas %>%
  mutate(across(corr, my_scale)) %>%
  ggplot(aes(x = corr, fill = false_negative)) +
  geom_density(alpha = 0.5)

dt_test_misclas %>%
  mutate(across(sd, ~ my_scale(log(.x)))) %>%
  ggplot(aes(x = sd, fill = false_negative)) +
  geom_density(alpha = 0.5)

dt_test_misclas %>%
  mutate(across(rad_af, my_scale)) %>%
  ggplot(aes(x = rad_af, fill = false_negative)) +
  geom_density(alpha = 0.5)
