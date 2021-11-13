# https://rspatial.org/terra/rs/2-exploration.html#relation-between-bands
library(tidyverse)
# library(terra)
library(ggcorrplot)
library(patchwork)

img1 <- read.table("data/imagem1.txt")
img2 <- read.table("data/imagem2.txt")
img3 <- read.table("data/imagem3.txt")


#  Data transformation ---------------------------------------------------

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

full_dt <- full_dt %>%
  filter(x > 69, x < 369)

# 1 Image summary ---------------------------------------------------------
# Labels
full_dt %>%
  count(img, label) %>%
  pivot_wider(names_from = img, values_from = n) %>%
  janitor::adorn_totals(where = "col", name = "Total") %>%
  mutate(across(img1:Total, ~paste0(round(.x/sum(.x)*100, 2), "%")))

colors <- c(
  "Cloud" = "white",
  "Unlabelled" = "black",
  "No cloud" = "gray"
)

labs_img1 <- full_dt %>%
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
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "none"
  )

labs_img2 <- full_dt %>%
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
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_manual(values = colors) +
  labs(x = "", y = "") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "none"
  )

labs_img3 <- full_dt %>%
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
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_manual(values = colors) +
  labs(x = "", y = "") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "none"
  )

ggsave(
  "graphs/01_labs_img1.png",
  labs_img1,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

ggsave(
  "graphs/01_labs_img2.png",
  labs_img2,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

ggsave(
  "graphs/01_labs_img3.png",
  labs_img3,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)


# 2 EDA -------------------------------------------------------------------
dt_eda <- full_dt %>%
  filter(label != 0)
# Authors claim radiance measures are not consistent
dt_eda %>%
  select(rad_df:img) %>%
  pivot_longer(cols = -img) %>%
  ggplot(aes(x = value, fill = img)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name)

my_scale <- function(x) {
  (x - mean(x)) / sd(x)
}

dt_eda %>%
  select(ndai:img) %>%
  mutate(across(-img, my_scale)) %>%
  pivot_longer(cols = -img) %>%
  ggplot(aes(x = value, fill = img)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name, scales = "free")

# Correlogram

corplot <- dt_eda %>%
  select(label:rad_an) %>%
  cor() %>%
  ggcorrplot(
    method = "square",
    type = "lower",
    ggtheme = ggplot2::theme_bw,
    legend.title = "Correlation"
  )

ggsave(
  "graphs/01_corrplot.png",
  corplot,
  width = 8*1.5,
  height = 6*1.5,
  units = "cm"
)

# Pairwise relationships
medians <- dt_eda %>%
  mutate(`log(sd)` = log(sd)) %>%
  select(label, ndai, corr, `log(sd)`, rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == 1, "Cloud", "No cloud"))) %>%
  group_by(label) %>%
  summarise(across(ndai:rad_an, median), .groups = "drop") %>%
  pivot_longer(cols = -label)

color_vec <- c("Cloud" = "gray60", "No cloud" = "gray8")

label_densities <- dt_eda %>%
  mutate(`log(sd)` = log(sd)) %>%
  select(label, ndai, corr, `log(sd)`, rad_an) %>%
  mutate(across(-label, my_scale)) %>%
  mutate(across(label, ~if_else(.x == 1, "Cloud", "No cloud"))) %>%
  pivot_longer(cols = -label) %>%
  ggplot(aes(x = value, fill = factor(label))) +
  geom_density(alpha = 0.7) +
  geom_vline(
    data = medians,
    aes(xintercept = value, color = factor(label)),
    linetype = 2
    ) +
  facet_wrap(~name, scales = "free") +
  labs(x = "", y = "", fill = "") +
  guides(color = "none") +
  # scale_fill_viridis_d(option = "plasma") +
  # scale_color_viridis_d(option = "plasma") +
  scale_fill_manual(values = color_vec) +
  scale_color_manual(values = color_vec) +
  theme_bw()

ggsave(
  "graphs/01_lab_densities.png",
  label_densities,
  width = 12*1.5,
  height = 6*1.5,
  units = "cm"
)

dt_eda %>%
  ggplot(aes(x = rad_af, y = rad_an)) +
  geom_point()

corr_radan <- dt_eda %>%
  ggplot(aes(x = corr, y = rad_an)) +
  geom_point(color = viridis::viridis(1)) +
  labs(x = "corr", y = "rad_an") +
  theme_bw()

ggsave(
  "graphs/01_corr_radan.png",
  corr_radan,
  width = 12*1.5,
  height = 6*1.5,
  units = "cm"
)

ndai_sd <- dt_eda %>%
  ggplot(aes(x = ndai, y = sd)) +
  geom_point(color = viridis::viridis(1)) +
  labs(x = "ndai", y = "sd") +
  theme_bw()

ggsave(
  "graphs/01_ndai_sd.png",
  ndai_sd,
  width = 12*1.5,
  height = 6*1.5,
  units = "cm"
)

dt_eda %>%
  ggplot(aes(x = corr)) +
  geom_density()


pca <- full_dt %>%
  select(-x, -y, -label, -img) %>%
  prcomp(scale. = TRUE)

cumsum(pca$sdev^2)/sum(pca$sdev^2)

pcimg <- as_tibble(pca$x) %>%
  mutate(x = full_dt$x, y = full_dt$y, img = full_dt$img)

pcimg %>%
  mutate(label = full_dt$label) %>%
  filter(label != 0) %>%
  group_by(label) %>%
  summarise(across(PC1:PC2, list("IQR" = IQR, "median" = median)))

pcimg1 <- pcimg %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "", fill = "PC1") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    # legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

pcimg2 <- pcimg %>%
  filter(img == "img2") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "", fill = "PC1") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    # legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

pcimg3 <- pcimg %>%
  filter(img == "img3") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c(breaks = seq(from = -8, to = 1, by = 3)) +
  labs(x = "", y = "", fill = "PC1") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    # legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(
  "graphs/01_pcimg1.png",
  pcimg1,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

ggsave(
  "graphs/01_pcimg2.png",
  pcimg2,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

ggsave(
  "graphs/01_pcimg3.png",
  pcimg3,
  width = 2.99 * 3,
  height = 3.82 * 3.5,
  units = "cm"
)

write_rds(full_dt, "data/01_dt_full.rds")
write_rds(dt_eda, "data/01_dt_eda.rds")
