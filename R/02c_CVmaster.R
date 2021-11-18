## Script: 02a_preparation.R
## Inputs:
## Outputs:
# CVmaster() to source in later scripts

library(tidymodels)
library(tidyverse)

#' Make Block Splits
#'
#' @param dt data frame with x,y columns
#' @param columns columns for block splits
#' @param rows rows for block splits
#'
#' @return Numeric vector with a block number for each row in dt
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

#' Make K-Means clustering splits
#'
#' @param dt data frame with x,y columns
#' @param k number of clusters to split into
#'
#' @return Numeric vector with a cluster number for each row in dt
make_clusters <- function(dt, k) {
  clust <- kmeans(
    select(dt, x, y),
    centers = k,
    iter.max = 1000L,
    nstart = 3L
  )

  return(clust$cluster)
}

#' Make Cross-Validation splits in block or cluster structure
#'
#' @param .dt data frame with x,y columns
#' @param .method splitting method ("block", "kmeans")
#' @param .columns if .method = "block", number of columns in the splits
#' @param .rows if .method = "block", number of rows in the splits
#' @param .k if .method = "kmeans", number of clusters
#'
#' @return An object of class c("vfold_cv", "rset") to use with tidymodels
#'   functions
make_cvsplits <- function(.dt, .method = "block", .columns = 3, .rows = 3, .k) {
  cv_att <- list(v = 0, strata = FALSE, repeats = 1)

  if (.method == "block") {
    out <- vector(mode = "list", length = .columns * .rows)
    names <- vector(mode = "character", length = .columns * .rows)
    blocks <- make_blocks(.dt, .columns, .rows)
    for (i in 1:(.columns * .rows)) {
      val_set <- which(blocks == i)
      train_set <- (1:NROW(.dt))[-val_set]
      out[[i]] <- rsample::make_splits(
        list(analysis = train_set, assessment = val_set),
        data = .dt
      )
      names[[i]] <- paste0("Fold", i)
    }
    cv_att$v <- .columns * .rows
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

#' Cross Validation model assessment
#'
#' @param .train data frame with training data
#' @param .classifier a tidymodels classification model
#' @param .recipe a tidymodels recipe
#' @param .method splitting method ("block", "kmeans")
#' @param .columns if .method = "block", number of columns in the splits
#' @param .rows if .method = "block", number of rows in the splits
#' @param .k if .method = "kmeans", number of clusters
#' @param .metrics a set of metrics specified with metric_set(). Some examples
#'   for classification are accuracy, roc_auc, mn_log_loss, sensitivity,
#'   precision
#'
#' @return A tibble with the metrics obtained for each fold
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
