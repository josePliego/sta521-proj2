# https://rspatial.org/terra/rs/2-exploration.html#relation-between-bands
library(tidyverse)
library(terra)

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
# Lose 2902/345556

# Make images square
# full_dt %>%
#   filter(img == "img1") %>%
#   select(x, y, label) %>%
#   pivot_wider(names_from = x, values_from = label) %>%
#   select(`65`:`100`)

# full_dt <- full_dt %>%
#   filter(!(img == "img1" & x <= 69))
# Lose 892 rows from image 1 (892/115110)

# full_dt %>%
#   filter(img == "img2") %>%
#   select(x, y, label) %>%
#   pivot_wider(names_from = x, values_from = label) %>%
#   select(`65`:`100`)

# full_dt <- full_dt %>%
#   filter(!(img == "img2" & x <= 69)) %>%
# Lose 790 rows from image 2 (790/115229)

# full_dt %>%
#   filter(img == "img3") %>%
#   select(x, y, label) %>%
#   pivot_wider(names_from = x, values_from = label) %>%
#   select(`65`:`100`)

# full_dt <- full_dt %>%
#   filter(!(img == "img3" & x <= 69))
# Lose 755 rows from image 3 (755/115217)

# 1 Image summary ---------------------------------------------------------
# Labels
full_dt %>%
  count(img, label) %>%
  pivot_wider(names_from = img, values_from = n) %>%
  janitor::adorn_totals(where = "col", name = "Total") %>%
  mutate(across(img1:Total, ~paste0(round(.x/sum(.x)*100, 2), "%")))

# Maps
# colors <- c(
#   "Cloud" = viridis(3)[[1]],
#   "Unlabelled" = viridis(3)[[2]],
#   "No cloud" = viridis(3)[[3]]
# )

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
  select(rad_df:img) %>%
  mutate(across(-img, my_scale)) %>%
  pivot_longer(cols = -img) %>%
  ggplot(aes(x = value, fill = img)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name)

pca <- dt_eda %>%
  select(-x, -y, -label, -img) %>%
  prcomp(scale. = TRUE)

pcimg1 <- as_tibble(pca$x) %>%
  mutate(x = dt_eda$x, y = dt_eda$y, img = dt_eda$img) %>%
  filter(img == "img1") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

pcimg2 <- as_tibble(pca$x) %>%
  mutate(x = dt_eda$x, y = dt_eda$y, img = dt_eda$img) %>%
  filter(img == "img2") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c() +
  labs(x = "", y = "") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

pcimg3 <- as_tibble(pca$x) %>%
  mutate(x = dt_eda$x, y = dt_eda$y, img = dt_eda$img) %>%
  filter(img == "img3") %>%
  ggplot(aes(x = x, y = y, color = PC1)) +
  geom_point() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_viridis_c(breaks = seq(from = -8, to = 1, by = 3)) +
  labs(x = "", y = "") +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_blank(),
    legend.position = "top",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(
  "graphs/01_pcimg1.png",
  pcimg1,
  width = 2.99 * 2,
  height = 3.82 * 2.5,
  units = "cm"
)

ggsave(
  "graphs/01_pcimg2.png",
  pcimg2,
  width = 2.99 * 2,
  height = 3.82 * 2.5,
  units = "cm"
)

ggsave(
  "graphs/01_pcimg3.png",
  pcimg3,
  width = 2.99 * 2,
  height = 3.82 * 2.5,
  units = "cm"
)

# dt_vectors <- as_tibble(pca$rotation) %>%
#   mutate(variable = rownames(pca$rotation))

# as_tibble(pca$x) %>%
#   mutate(label = dt_eda$label) %>%
#   ggplot() +
#   geom_point(aes(x = PC1, y = PC2, color = factor(label)), alpha = 0.2) +
#   geom_segment(
#     aes(x = 0, y = 0, xend = 10*PC1, yend = 10*PC2),
#     data = dt_vectors
#     )
