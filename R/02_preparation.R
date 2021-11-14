library(tidyverse)
dt_full <- read_rds("data/01_dt_full.rds")

# 1 Split -----------------------------------------------------------------
dt_img1 <- dt_full %>%
  filter(img == "img1")
dt_img2 <- dt_full %>%
  filter(img == "img2")
dt_img3 <- dt_full %>%
  filter(img == "img3")

# Block split
make_blocks <- function(dt, columns, rows) {

  if (rows > 1) {
    cut_y1 <- cut(dt$y, breaks = rows, labels = FALSE)
  } else {
    cut_y1 <- rep(1L, times = length(dt$y))
  }

  if (columns > 1) {
    cut_x1 <- cut(dt$x, breaks = columns, labels = FALSE)
  } else {
    cut_x1 <- rep(1L, times = length(dt$x))
  }

  aux <- dt %>%
    mutate(y_group = cut_y1, x_group = cut_x1, block = NA_integer_)

  y_levels <- unique(cut_y1)
  x_levels <- unique(cut_x1)
  indicator <- 0L

  for (i in seq_along(y_levels)) {
    for (j in seq_along(x_levels)) {
      indicator <- indicator + 1L
      aux <- aux %>%
        mutate(block = case_when(
          y_group == y_levels[[i]] & x_group == x_levels[[j]] ~ indicator,
          TRUE ~ block
        ))
    }
  }
  aux <- select(aux, -x_group, -y_group)
  return(aux)

}

blocks1 <- make_blocks(dt_img1, 2, 1)

blocks1 %>%
  filter(label != 0) %>%
  group_by(block) %>%
  summarise(a = list("mean" = mean(label), "n" = n()), .groups = "drop") %>%
  unnest_wider(a)

blocks2 <- make_blocks(dt_img2, 2, 1)

blocks2 %>%
  filter(label != 0) %>%
  group_by(block) %>%
  summarise(a = list("mean" = mean(label), "n" = n()), .groups = "drop") %>%
  unnest_wider(a)

blocks3 <- make_blocks(dt_img3, 2, 1)

blocks3 %>%
  filter(label != 0) %>%
  group_by(block) %>%
  summarise(a = list("mean" = mean(label), "n" = n()), .groups = "drop") %>%
  unnest_wider(a)

dt_full %>%
  filter(label != 0) %>%
  NROW()

dt_blocks <- bind_rows(blocks1, blocks2, blocks3)

set.seed(42)
img_test <- sample(c("img1", "img2", "img3"), 1)
block_test <- sample(c(1, 2), 1)
img_val <- sample(c("img1", "img2", "img3"), 1)
block_val <- sample(c(1, 2), 1)
while (img_val == img_test & block_val == block_test) {
  img_val <- sample(c("img1", "img2", "img3"), 1)
  block_val <- sample(c(1, 2), 1)
}

dt_test_block <- dt_blocks %>%
  filter(img == img_test, block == block_test)
dt_val_block <- dt_blocks %>%
  filter(img == img_val, block == block_val)
dt_train_block <- dt_blocks %>%
  filter(
    !(img == img_test & block == block_test),
    !(img == img_val & block == block_val)
    )

blocks_img3 <- dt_blocks %>%
  filter(img == "img3", label != 0) %>%
  ggplot(aes(x = x, y = y, color = factor(block))) +
  geom_point() +
  scale_color_viridis_d() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  "graphs/02_blocksimg3.png",
  blocks_img3,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

# Kmeans split
my_scale <- function(x) {
  (x - mean(x)) / sd(x)
}

clust_img1 <- kmeans(
  dt_img1 %>%
    filter(label != 0) %>%
    select(x, y) %>%
    mutate(across(c(x, y), my_scale)),
  centers = 3,
  iter.max = 1e5,
  nstart = 5
  )

clust_img2 <- kmeans(
  dt_img2 %>%
    filter(label != 0) %>%
    select(x, y) %>%
    mutate(across(c(x, y), my_scale)),
  centers = 3,
  iter.max = 1e5,
  nstart = 5
)

clust_img3 <- kmeans(
  dt_img3 %>%
    filter(label != 0) %>%
    select(x, y) %>%
    mutate(across(c(x, y), my_scale)),
  centers = 3,
  iter.max = 1e5,
  nstart = 5
)

# dt_img1 %>%
#   filter(label != 0) %>%
#   mutate(cluster = clust_img1$cluster) %>%
#   ggplot(aes(x = x, y = y, color = cluster)) +
#   geom_point()

# dt_img2 %>%
#   filter(label != 0) %>%
#   mutate(cluster = clust_img2$cluster) %>%
#   ggplot(aes(x = x, y = y, color = cluster)) +
#   geom_point()

kmeans_img3 <- dt_img3 %>%
  filter(label != 0) %>%
  mutate(cluster = clust_img3$cluster) %>%
  ggplot(aes(x = x, y = y, color = factor(cluster))) +
  geom_point() +
  scale_color_viridis_d() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  "graphs/02_kmeansimg3.png",
  kmeans_img3,
  width = 2.99 * 2,
  height = 3.82 * 2,
  units = "cm"
)

kmeans1 <- dt_img1 %>%
  filter(label != 0) %>%
  mutate(cluster = clust_img1$cluster)

kmeans2 <- dt_img2 %>%
  filter(label != 0) %>%
  mutate(cluster = clust_img2$cluster) %>%
  mutate(across(cluster, ~.x + 3))

kmeans3 <- dt_img2 %>%
  filter(label != 0) %>%
  mutate(cluster = clust_img2$cluster) %>%
  mutate(across(cluster, ~.x + 6))

set.seed(42)
dt_kmeans <- bind_rows(kmeans1, kmeans2, kmeans3)
clust_test <- sample(1:9, size = 2, replace = FALSE)
clust_val <- sample(setdiff(1:9, clust_test), size = 1)

dt_test_kmeans <- dt_kmeans %>%
  filter(cluster %in% clust_test)
dt_val_kmeans <- dt_kmeans %>%
  filter(cluster %in% clust_val)
dt_train_kmeans <- dt_kmeans %>%
  filter(!cluster %in% clust_test, !cluster %in% clust_val)


# 2 Trivial classifier ----------------------------------------------------

dt_test_block %>%
  filter(label != 0) %>%
  summarise(acc = mean(label == -1))

dt_val_block %>%
  filter(label != 0) %>%
  summarise(acc = mean(label == -1))

dt_test_kmeans %>%
  summarise(acc = mean(label == -1))

dt_val_kmeans %>%
  summarise(acc = mean(label == -1))
