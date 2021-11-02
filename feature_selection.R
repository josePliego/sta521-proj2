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
