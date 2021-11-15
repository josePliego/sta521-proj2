source("R/02_CVmaster.R")

dt_full <- read_rds("data/01_dt_full.rds") %>%
  filter(label != 0) %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x))) %>%
  mutate(across(label, factor))

dt_train <- read_rds("data/02_dt_train_block.rds") %>%
  mutate(
    across(x, ~if_else(img == "img3", .x + 299, .x))
  ) %>%
  mutate(across(label, factor))

dt_rec <- recipe(label ~., data = dt_train) %>%
  step_rm(x, y, img) %>%
  step_scale(all_predictors())

logreg_mod <- logistic_reg(mode = "classification", engine = "glm")

CVmaster(
  dt_train,
  logreg_mod,
  dt_rec,
  .method = "kmeans",
  .k = 5,
  .metrics = metric_set(accuracy, mn_log_loss, kap, yardstick::spec, sens)
)
