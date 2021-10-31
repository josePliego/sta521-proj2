# 1 Preamble --------------------------------------------------------------

library(tidyverse)
library(viridis)
library(patchwork)
library(factoextra)

img1 <- read.table("data/imagem1.txt")
img2 <- read.table("data/imagem2.txt")
img3 <- read.table("data/imagem3.txt")


# 2 Data transformation ---------------------------------------------------

column_names <- c(
  "y",
  "x",
  "label",
  "ndai",
  "sd",
  "corr",
  "rad_df",
  "rad_cf",
  "rad_bf",
  "rad_af",
  "rad_an"
  )
names(img1) <- column_names
names(img2) <- column_names
names(img3) <- column_names


full_dt <- bind_rows(
  mutate(tibble(img1), img = "img1"),
  mutate(tibble(img2), img = "img2"),
  mutate(tibble(img3), img = "img3")
)

full_dt %>% write_rds("data/full_dt.rds")


# 3 EDA -------------------------------------------------------------------
# See number of unlabelled points
full_dt %>%
  count(factor(label)) %>%
  mutate(prop = n/sum(n))

full_dt %>%
  group_by(img) %>%
  count(label) %>%
  mutate(prop = n/sum(n)) %>%
  ungroup()


# 3.1 Image plots ---------------------------------------------------------

# colors <- c(
#   "Cloud" = "whitesmoke", "Unlabelled" = "gray47", "No cloud" = "powderblue"
#   )

colors <- c(
  "Cloud" = viridis(3)[[1]],
  "Unlabelled" = viridis(3)[[2]],
  "No cloud" = viridis(3)[[3]]
)

full_dt %>%
  filter(img == "img1") %>%
  mutate(
    across(
      label,
      ~case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled"
      )
    )
    ) %>%
  ggplot(aes(x = x, y = y, color = factor(label))) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_manual(values = colors) +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank()
  )

full_dt %>%
  filter(img == "img2") %>%
  mutate(
    across(
      label,
      ~case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled"
      )
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = factor(label))) +
  geom_point() +
  scale_color_manual(values = colors) +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank()
  )

full_dt %>%
  filter(img == "img3") %>%
  mutate(
    across(
      label,
      ~case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled"
      )
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = factor(label))) +
  geom_point() +
  scale_color_manual(values = colors) +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank()
  )


# 3.2 PCA -----------------------------------------------------------------

pca_full <- full_dt %>%
  select(-label, -img, -x, -y) %>%
  prcomp(scale. = TRUE)

fviz_screeplot(pca_full)
p1 <- fviz_pca_var(pca_full)
# fviz_pca_biplot(pca_full)

p2 <- as_tibble(pca_full$x) %>%
  mutate(
    img = full_dt$img,
    label = full_dt$label
    ) %>%
  filter(img == "img1") %>%
  mutate(
    across(
      label,
      ~case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled"
      )
    )
  ) %>%
  ggplot(aes(x = PC1, y = PC2, color = factor(label))) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_manual(values = colors) +
  theme_minimal()

p1|p2


# 3.3 Predictors in space -------------------------------------------------

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = rad_df)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = rad_cf)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = rad_bf)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = rad_af)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = rad_an)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = ndai)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = sd)) +
  geom_point()

full_dt %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = corr)) +
  geom_point()


# 3.4 PC Images -----------------------------------------------------------

pca_img1 <- full_dt %>%
  filter(img == "img1") %>%
  select(-label, -img, -x, -y) %>%
  prcomp(scale. = TRUE)

as_tibble(pca_img1$x) %>%
  mutate(
    label = full_dt %>% filter(img == "img1") %>% .$label,
    x = full_dt %>% filter(img == "img1") %>% .$x,
    y = full_dt %>% filter(img == "img1") %>% .$y
    ) %>%
  mutate(
    across(
      label,
      ~case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled"
      )
    )
  ) %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point()



# 3.5 Predictors by label -------------------------------------------------

full_dt %>%
  filter(img == "img1", x >= 70) %>%
  select(-img) %>%
  pivot_longer(cols = c(-x, -y, -label)) %>%
  filter(label != 0) %>%
  ggplot(aes(x = value, fill = factor(label))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name, scales = "free")


# 3.6 Author defined variables --------------------------------------------

full_dt %>%
  filter(img == "img1", x >= 70) %>%
  filter(label != 0) %>%
  ggplot(aes(x = corr, y = sd, color = factor(label))) +
  geom_point()

full_dt %>%
  filter(img == "img1", x >= 70) %>%
  filter(label != 0) %>%
  ggplot(aes(x = corr, y = ndai, color = factor(label))) +
  geom_point()

full_dt %>%
  filter(img == "img1", x >= 70) %>%
  filter(label != 0) %>%
  ggplot(aes(x = ndai, y = sd, color = factor(label))) +
  geom_point()

plotly::plot_ly(
  x = full_dt %>% filter(img == "img1", x >= 70, label != 0) %>% .$corr,
  y = full_dt %>% filter(img == "img1", x >= 70, label != 0) %>% .$ndai,
  z = full_dt %>% filter(img == "img1", x >= 70, label != 0) %>% .$sd,
  type = "scatter3d",
  color = full_dt %>% filter(img == "img1", x >= 70, label != 0) %>% .$label %>% factor(),
  alpha = 0.5,
  mode = "markers",
  size = I(3)
)
