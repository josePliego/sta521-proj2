library(tidyverse)
library(OpenImageR)

dt_train <- read_rds("data/02_dt_train_block.rds") %>%
  select(-block) %>%
  mutate(
    across(x, ~if_else(img == "img3", .x + 299, .x))
  )

dt_pca <- dt_train %>%
  select(ndai:rad_an) %>%
  prcomp(scale. = TRUE) %>%
  .$x %>%
  as_tibble() %>%
  mutate(x = dt_train$x, y = dt_train$y) %>%
  pivot_wider(names_from = x) %>%
  select(-y) %>%
  as.matrix()

init_gb <- GaborFeatureExtract$new()

dat <- init_gb$gabor_feature_engine(
  img_data = dt_pca, img_nrow = NROW(dt_pca), img_ncol = NCOL(dt_pca),
  scales = 4, orientations = 8, gabor_rows = 39,
  gabor_columns = 39, downsample_gabor = FALSE,
  downsample_rows = NULL, downsample_cols = NULL,
  normalize_features = FALSE, threads = 6,
  verbose = TRUE)



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
