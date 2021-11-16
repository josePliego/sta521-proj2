# devtools::install_github("tidymodels/parsnip")
# devtools::install_github("tidymodels/discrim")
library(tidymodels)
library(tidyverse)
library(discrim)
library(MASS, pos = 999)

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

wf_logreg <- workflow() %>%
  add_model(mod_logreg) %>%
  add_recipe(rec_logreg)

fit_logreg <- wf_logreg %>%
  fit(dt_train_logreg)

roc_logreg <- fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_curve(label, .pred_0)

fit_logreg %>%
  augment(new_data = dt_train_logreg) %>%
  roc_auc(label, .pred_0)

roc_logreg %>%
  autoplot() +
  geom_point(
    data = roc_logreg %>% filter(abs(.threshold - 0.5)<0.0001) %>% slice(1),
    aes(x = 1 - specificity, y = sensitivity),
    color = "red",
    size = 1
  )
