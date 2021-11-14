library(tidyverse)
library(modelr)
library(yardstick)
library(xtable)

my_scale <- function(x) {
  (x - mean(x)) / sd(x)
}

dt_train_block <- read_rds("data/02_dt_train_block.rds") %>%
  select(label:rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x)))

dt_val_block <- read_rds("data/02_dt_val_block.rds") %>%
  select(label:rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x)))

dt_train_kmeans <- read_rds("data/02_dt_train_kmeans.rds") %>%
  select(label:rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x)))

dt_val_kmeans <- read_rds("data/02_dt_val_kmeans.rds") %>%
  select(label:rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == -1, 0, .x)))

make_model <- function(var, dt) {
  fmla <- as.formula(paste0("label ~ ", var))
  glm(fmla, data = dt, family = "binomial")
}

get_roc_auc <- function(model, dt) {
  dt %>%
    mutate(
      estimate = predict(model, newdata = dt, type = "response")
    ) %>%
    mutate(across(label, factor)) %>%
    roc_auc(estimate, truth = label, event_level = "second") %>%
    .$`.estimate`
}

plot_roc_curve <- function(model, dt) {
  dt %>%
    mutate(
      estimate = predict(model, newdata = dt, type = "response")
    ) %>%
    mutate(across(label, factor)) %>%
    roc_curve(estimate, truth = label, event_level = "second") %>%
    autoplot()
}

vars <- c(
  "ndai", "sd", "corr", "rad_df", "rad_cf", "rad_bf", "rad_af", "rad_an"
  )

dt_models <- tibble::tibble(
  "var" = vars
) %>%
  mutate(
    model_block = map(var, ~make_model(.x, dt_train_block)),
    model_kmeans = map(var, ~make_model(.x, dt_train_kmeans))
    )

dt_roc_auc <- dt_models %>%
  mutate(
    auc_block = map_dbl(model_block, ~get_roc_auc(.x, dt_val_block)),
    auc_block_plot = map(model_block, ~plot_roc_curve(.x, dt_val_block)),
    auc_kmeans = map_dbl(model_kmeans, ~get_roc_auc(.x, dt_val_kmeans)),
    auc_kmeans_plot = map(model_kmeans, ~plot_roc_curve(.x, dt_val_kmeans))
  )

dt_roc_auc %>%
  select(var, auc_block, auc_kmeans, everything()) %>%
  mutate(mean_auc = (auc_block + auc_kmeans)/2) %>%
  select(var, auc_block:auc_kmeans, mean_auc) %>%
  arrange(-mean_auc) %>%
  xtable()

dt_full <- read_rds("data/01_dt_full.rds")
ndai1 <- dt_full %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = ndai)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "", fill = "ndai") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ndai2 <- dt_full %>%
  filter(img == "img2") %>%
  ggplot(aes(x = x, y = y, color = ndai)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "", fill = "ndai") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ndai3 <- dt_full %>%
  filter(img == "img3") %>%
  ggplot(aes(x = x, y = y, color = ndai)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "", fill = "ndai") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(
  "graphs/02_ndaiimg1.png",
  ndai1,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

ggsave(
  "graphs/02_ndaiimg2.png",
  ndai2,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

ggsave(
  "graphs/02_ndaiimg3.png",
  ndai3,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)
