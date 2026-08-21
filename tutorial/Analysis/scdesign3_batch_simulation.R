library(Seurat)
library(scDesign3)
library(SingleCellExperiment)
library(zellkonverter)
library(ggplot2)

input_dir <- "D:/SpaDiff/0_data/1_DLPFC"
output_dir <- "D:/SpaDiff/0_data/donor3_151673_151676"
slice_ids <- c("151673", "151674", "151675", "151676")
simulation_seed <- 0803
n_cores <- 16

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj_list <- lapply(seq_along(slice_ids), function(i) {
  slice_id <- slice_ids[i]
  slice_dir <- file.path(input_dir, slice_id)
  obj_i <- Load10X_Spatial(slice_dir)

  truth <- read.delim(
    file.path(slice_dir, "truth.txt"),
    sep = "\t",
    header = FALSE,
    row.names = 1,
    stringsAsFactors = FALSE
  )
  obj_i$batch <- paste0("batch", i)
  obj_i$truth <- as.character(truth[colnames(obj_i), 1])
  labelled_cells <- colnames(obj_i)[!is.na(obj_i$truth) & obj_i$truth != ""]
  obj_i <- subset(obj_i, cells = labelled_cells)

  coordinates <- GetTissueCoordinates(obj_i[["slice1"]])
  if ("cell" %in% colnames(coordinates)) {
    rownames(coordinates) <- coordinates$cell
  }
  coordinates <- coordinates[colnames(obj_i), , drop = FALSE]
  obj_i$spatial1 <- coordinates$y
  obj_i$spatial2 <- coordinates$x
  obj_i
})

obj <- merge(
  obj_list[[1]],
  y = obj_list[-1],
  add.cell.ids = seq_along(obj_list)
)
obj <- JoinLayers(obj, assay = DefaultAssay(obj))

feature_list <- lapply(obj_list, function(x) {
  VariableFeatures(FindVariableFeatures(x, nfeatures = 8000))
})
features <- Reduce(intersect, feature_list)
sce <- as.SingleCellExperiment(obj)[features, ]

BATCH_data <- construct_data(
  sce = sce,
  assay_use = "counts",
  celltype = "truth",
  pseudotime = NULL,
  spatial = c("spatial1", "spatial2"),
  other_covariates = "batch",
  corr_by = "1"
)
BATCH_marginal <- fit_marginal(
  data = BATCH_data,
  predictor = "gene",
  mu_formula = "truth + batch",
  sigma_formula = "1",
  family_use = "nb",
  n_cores = n_cores,
  usebam = FALSE
)

valid_gene <- vapply(
  BATCH_marginal,
  function(x) length(class(x$fit)) > 1,
  logical(1)
)
sce <- sce[valid_gene, ]

BATCH_data <- construct_data(
  sce = sce,
  assay_use = "counts",
  celltype = "truth",
  pseudotime = NULL,
  spatial = c("spatial1", "spatial2"),
  other_covariates = "batch",
  corr_by = "1"
)
BATCH_marginal <- fit_marginal(
  data = BATCH_data,
  predictor = "gene",
  mu_formula = "truth + batch",
  sigma_formula = "1",
  family_use = "nb",
  n_cores = n_cores,
  usebam = FALSE
)
BATCH_copula <- fit_copula(
  sce = sce,
  assay_use = "counts",
  marginal_list = BATCH_marginal,
  family_use = "nb",
  copula = "gaussian",
  n_cores = 1,
  input_data = BATCH_data$dat
)

batch_strength <- data.frame(
  level = c("low", "mid", "high"),
  mean = c(0.20, 0.40, 0.60),
  sd = c(0.10, 0.15, 0.25)
)
write.csv(
  batch_strength,
  file.path(output_dir, "batch_strength.csv"),
  row.names = FALSE
)

for (strength_i in seq_len(nrow(batch_strength))) {
  level <- batch_strength$level[strength_i]
  level_dir <- file.path(output_dir, level)
  dir.create(level_dir, recursive = TRUE, showWarnings = FALSE)

  set.seed(simulation_seed)
  BATCH_marginal_alter <- lapply(BATCH_marginal, function(x) {
    batch_coef <- grep("^batch", names(x$fit$coefficients))
    effect_direction <- sample(c(-1, 1), length(batch_coef), replace = TRUE)
    effect_size <- abs(rnorm(
      length(batch_coef),
      mean = batch_strength$mean[strength_i],
      sd = batch_strength$sd[strength_i]
    ))
    x$fit$coefficients[batch_coef] <- effect_direction * effect_size
    x
  })

  injected_batch_effect <- do.call(rbind, lapply(
    seq_along(BATCH_marginal_alter),
    function(i) {
      coefficients_i <- BATCH_marginal_alter[[i]]$fit$coefficients
      batch_coef <- grep("^batch", names(coefficients_i))
      data.frame(
        gene = rownames(sce)[i],
        batch_term = names(coefficients_i)[batch_coef],
        coefficient = unname(coefficients_i[batch_coef]),
        fold_change = exp(unname(coefficients_i[batch_coef]))
      )
    }
  ))
  write.csv(
    injected_batch_effect,
    file.path(level_dir, "injected_batch_effect.csv"),
    row.names = FALSE
  )

  BATCH_para_alter <- extract_para(
    sce = sce,
    marginal_list = BATCH_marginal_alter,
    n_cores = 1,
    family_use = "nb",
    new_covariate = BATCH_data$newCovariate,
    data = BATCH_data$dat
  )
  simulated_counts <- simu_new(
    sce = sce,
    mean_mat = BATCH_para_alter$mean_mat,
    sigma_mat = BATCH_para_alter$sigma_mat,
    zero_mat = BATCH_para_alter$zero_mat,
    quantile_mat = NULL,
    copula_list = BATCH_copula$copula_list,
    n_cores = n_cores,
    family_use = "nb",
    input_data = BATCH_data$dat,
    new_covariate = BATCH_data$newCovariate,
    important_feature = BATCH_copula$important_feature,
    filtered_gene = BATCH_data$filtered_gene
  )

  simulated_object <- CreateSeuratObject(
    counts = simulated_counts,
    meta.data = BATCH_data$newCovariate
  )
  simulated_object <- NormalizeData(simulated_object)
  simulated_object <- FindVariableFeatures(simulated_object, nfeatures = 2000)
  simulated_object <- ScaleData(simulated_object)
  simulated_object <- RunPCA(simulated_object, verbose = FALSE)
  simulated_object <- RunUMAP(
    simulated_object,
    reduction = "pca",
    dims = 1:30
  )

  batch_umap <- DimPlot(simulated_object, group.by = "batch")
  truth_umap <- DimPlot(simulated_object, group.by = "truth")
  ggsave(file.path(level_dir, "batch.pdf"), batch_umap)
  ggsave(file.path(level_dir, "cell_type.pdf"), truth_umap)

  simulated_sce <- as.SingleCellExperiment(simulated_object)
  writeH5AD(
    simulated_sce,
    file.path(level_dir, "simulated_data.h5ad")
  )
}
