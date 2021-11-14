library(tidyverse)
library(philentropy)

full_dt <- read_rds("data/full_dt.rds")

get_kl <- function(dt, var) {
  label_cloud <- dt %>%
    filter(label == 1) %>%
    .[[var]]

  label_nocloud <- dt %>%
    filter(label == -1) %>%
    .[[var]]

  min_dens <- min(
    dt %>%
      filter(label != 0) %>%
      .[[var]]
  )

  max_dens <- max(
    dt %>%
      filter(label != 0) %>%
      .[[var]]
  )

  no_cloud <- density(label_nocloud, from = min_dens, to = max_dens)
  cloud <- density(label_cloud, from = min_dens, to = max_dens)

  kl1 <- KL(
    rbind(no_cloud$y/sum(no_cloud$y), cloud$y/sum(cloud$y)),
    unit = "log2"
    )
  kl2 <- KL(
    rbind(cloud$y/sum(cloud$y), no_cloud$y/sum(no_cloud$y)),
    unit = "log2"
    )

  return(kl1 + kl2)

}

dt_split <- full_dt %>%
  select(-x, -y) %>%
  split(.$img)

vars <- c("ndai", "sd", "corr", "rad_df", "rad_cf", "rad_bf", "rad_af", "rad_an")

output <- tibble(
  img = rep("img", times = 24),
  var = rep("var", times = 24),
  kl = rep(0, times = 24)
)

for (i in 1:3) {
  for (j in seq_along(vars)) {
    output[8*(i-1) + j, 1] <- paste0("img", i)
    output[8*(i-1) + j, 2] <- vars[[j]]
    output[8*(i-1) + j, 3] <- get_kl(dt_split[[i]], vars[[j]])
  }
}

my_scale <- function(x) {
  (x - mean(x)) / sd(x)
}

img1 <- full_dt %>%
  filter(label != 0, img == "img1") %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x))) %>%
  select(-x, -y, -img) %>%
  mutate(across(-label, my_scale))

img2 <- full_dt %>%
  filter(label != 0, img == "img2") %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x))) %>%
  select(-x, -y, -img) %>%
  mutate(across(-label, my_scale))

img3 <- full_dt %>%
  filter(label != 0, img == "img3") %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x))) %>%
  select(-x, -y, -img) %>%
  mutate(across(-label, my_scale))

library(modelr)
library(magrittr)

make_model <- function(var, dt) {
  fmla <- as.formula(paste0("label ~ ", var))
  glm(fmla, data = eval(parse(text = dt)), family = "binomial")
}

get_accuracy <- function(model, dt) {
  dt <- eval(parse(text = dt))
  dt %>%
    add_predictions(model, type = "response") %>%
    mutate(across(pred, ~if_else(.x > 0.5, 1, 0))) %>%
    summarise(accuracy = mean(pred == label)) %>%
    .$accuracy
}

accuracies <- tibble(
  "feature" = rep(vars, times = 3),
  "img" = c(
    rep("img1", times = length(vars)),
    rep("img2", times = length(vars)),
    rep("img3", times = length(vars))
    )
  ) %>%
  mutate(model = map2(feature, img, ~make_model(.x, .y))) %>%
  mutate(accuracy = map2_dbl(model, img, ~get_accuracy(.x, .y)))

accuracies %>%
  group_by(feature) %>%
  summarise(across(accuracy, mean))

model_ndai <- glm(label ~ ndai, data = img1, family = "binomial")
model_corr <- glm(label ~ corr, data = img1, family = "binomial")

img1 %>%
  add_predictions(model_ndai, type = "response") %>%
  mutate(across(pred, ~if_else(.x > 0.5, 1, 0))) %>%
  summarise(accuracy = mean(pred == label))

img1 %>%
  add_predictions(model_corr, type = "response") %>%
  mutate(across(pred, ~if_else(.x > 0.5, 1, 0))) %>%
  summarise(accuracy = mean(pred == label))
