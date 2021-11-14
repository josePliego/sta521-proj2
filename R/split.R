library(tidyverse)

full_dt <- read_rds("data/full_dt.rds")

set.seed(42)
validation <- sample(1:6, size = 1)
test <- sample(setdiff(1:6, validation), size = 1)
train <- setdiff(1:6, c(validation, test))

splits <- full_dt %>%
  group_by(img) %>%
  mutate(
    img_split = if_else(x >= mean(x), "1", "0")
  ) %>%
  ungroup() %>%
  mutate(
    split = case_when(
      img == "img1" & img_split == "0" ~ 1,
      img == "img1" & img_split == "1" ~ 2,
      img == "img2" & img_split == "0" ~ 3,
      img == "img2" & img_split == "1" ~ 4,
      img == "img3" & img_split == "0" ~ 5,
      img == "img3" & img_split == "1" ~ 6
      )
  ) %>%
  mutate(
    set = case_when(
      split %in% validation ~ "validation",
      split %in% train ~ "train",
      split %in% test ~ "test"
    )
  ) %>%
  select(-img_split, -split)

## Exercise 2 part b

set.seed(10)

data_validation = splits %>%
  filter(set == "validation") %>%
  group_by(label) %>%
  summarise(n())

print(data_validation)

Perc_correct_val = 8493/(8493+34609)
Perc_incorrect_val = 1 - Perc_correct_val

data_test = splits %>%
  filter(set == "test") %>%
  group_by(label) %>%
  summarise(n())

print(data_validation)

Perc_correct_test = 28858/(28858+3208)
Perc_incorrect_test = 1 - Perc_correct_val

# Classifier will have high accuracy when test and validation sets have mostly cloud-less pixels.

