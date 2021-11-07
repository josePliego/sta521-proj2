# 1 Preamble --------------------------------------------------------------

library(tidyverse)
library(viridis)
library(patchwork)
library(factoextra)
library(OpenImageR)

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


# 3 Gabor features --------------------------------------------------------

img1_prcomp <- full_dt %>%
  filter(x >= 70) %>%
  filter(img == "img1") %>%
  select(-x, -y, -label, -img) %>%
  prcomp(scale. = TRUE)

init_gb <- GaborFeatureExtract$new()

img1_pca_matrix <- as_tibble(img1_prcomp$x) %>%
  mutate(
    x = full_dt %>% filter(x >= 70, img == "img1") %>% .$x,
    y = full_dt %>% filter(x >= 70, img == "img1") %>% .$y
  ) %>%
  arrange(x, y) %>%
  select(x, y, PC1) %>%
  pivot_wider(names_from = x, values_from = PC1) %>%
  select(-y) %>%
  as.matrix()

gb_im = init_gb$gabor_feature_extraction(
  image = img1_pca_matrix, scales = 5, orientations = 8,
  downsample_gabor = FALSE, downsample_rows = NULL,
  downsample_cols = NULL, gabor_rows = 31,
  gabor_columns = 31, plot_data = TRUE,
  normalize_features = FALSE, threads = 3, verbose = TRUE,
  vectorize_magnitude = TRUE
  )

plt_im_thresh = init_gb$plot_gabor(
  real_matrices = gb_im$gabor_features_real,
  margin_btw_plots = 0.65, thresholding = TRUE
  )

gabor_features <- tibble(
  x = full_dt %>% filter(x >= 70, img == "img1") %>% .$x,
  y = full_dt %>% filter(x >= 70, img == "img1") %>% .$y,
  label = full_dt %>% filter(x >= 70, img == "img1") %>% .$label
) %>%
  mutate(
    gabor1 = gb_im$gaborFeatures[[1]][1:nrow(.)],
    gabor2 = gb_im$gaborFeatures[[1]][(nrow(.)+1):(2*nrow(.))],
    gabor3 = gb_im$gaborFeatures[[1]][(2*nrow(.)+1):(3*nrow(.))],
    gabor4 = gb_im$gaborFeatures[[1]][(3*nrow(.)+1):(4*nrow(.))],
    gabor2 = gb_im$gaborFeatures[[1]][(4*nrow(.)+1):(5*nrow(.))]
    )

gabor_features %>%
  ggplot(aes(x = x, y = y, color = gabor2)) +
  geom_point()

gabor_features %>%
  filter(label != 0) %>%
  ggplot(aes(x = gabor1, fill = factor(label))) +
  geom_density(alpha = 0.5)

gabor_features %>%
  filter(label != 0) %>%
  ggplot(aes(x = gabor2, fill = factor(label))) +
  geom_density(alpha = 0.5)

gabor_features %>%
  pivot_longer(starts_with("gabor")) %>%
  filter(label != 0) %>%
  ggplot(aes(x = value, fill = factor(label))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name)
