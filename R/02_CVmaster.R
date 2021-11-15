library(tidymodels)
library(tidyverse)

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
  return(aux$block)

}

make_clusters <- function(dt, k) {
  clust <- kmeans(
    select(dt, x, y),
    centers = k,
    iter.max = 1000L,
    nstart = 3L
    )

  return(clust$cluster)

}

make_cvsplits <- function(.dt, .method = "block", .columns = 3, .rows = 3, .k) {
  cv_att <- list(v = 0, strata = FALSE, repeats = 1)

  if (.method == "block") {
    out <- vector(mode = "list", length = .columns*.rows)
    names <- vector(mode = "character", length = .columns*.rows)
    blocks <- make_blocks(.dt, .columns, .rows)
    for (i in 1:(.columns*.rows)) {
      val_set <- which(blocks == i)
      train_set <- (1:NROW(.dt))[-val_set]
      out[[i]] <- rsample::make_splits(
        list(analysis = train_set, assessment = val_set),
        data = .dt
        )
      names[[i]] <- paste0("Fold", i)
    }
    cv_att$v <- .columns*.rows
  }

  if (.method == "kmeans") {
    out <- vector(mode = "list", length = .k)
    names <- vector(mode = "character", length = .k)
    clusters <- make_clusters(.dt, .k)
    for (i in 1:.k) {
      val_set <- which(clusters == i)
      train_set <- (1:NROW(.dt))[-val_set]
      out[[i]] <- rsample::make_splits(
        list(analysis = train_set, assessment = val_set),
        data = .dt
      )
      names[[i]] <- paste0("Fold", i)
    }

    cv_att$v <- .k
  }

  rset_obj <- new_rset(
    splits = out,
    ids = names,
    subclass = c("vfold_cv", "rset"),
    attrib = cv_att
    )

  return(rset_obj)

}

CVmaster <- function(.train, .classifier, .recipe, .method, .columns,
                     .rows, .k, .metrics = metric_set(accuracy)) {

  fit <- workflow() %>%
    add_recipe(.recipe) %>%
    add_model(.classifier) %>%
    fit_resamples(
      make_cvsplits(.train, .method, .columns, .rows, .k),
      metrics = .metrics
      )

  return(select(bind_rows(fit$.metrics), -.config))

}
