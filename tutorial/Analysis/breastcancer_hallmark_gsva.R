library(clusterProfiler)
library(GSVA)
library(pheatmap)
library(RColorBrewer)


# 1. File paths and analysis parameters

expression_file <- "data/gene_exp_counts_sum.csv"
hallmark_gmt_file <- "data/h.all.v2025.1.Hs.symbols.gmt"
output_dir <- "results/hallmark_gsva"

replicates_per_group <- 3
group_names <- as.character(0:6)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Read the expression matrix
expression <- read.csv(
  expression_file,
  row.names = 1,
  check.names = FALSE
)
expression <- as.matrix(expression)
storage.mode(expression) <- "numeric"

# 3. Sum every three replicate columns into one experimental group
grouped_expression <- vapply(
  seq_along(group_names),
  function(group_index) {
    first_column <- (group_index - 1) * replicates_per_group + 1
    last_column <- group_index * replicates_per_group
    rowSums(expression[, first_column:last_column, drop = FALSE])
  },
  numeric(nrow(expression))
)

rownames(grouped_expression) <- rownames(expression)
colnames(grouped_expression) <- group_names

# 4. Read the MSigDB Hallmark gene sets
hallmark_table <- read.gmt(hallmark_gmt_file)

# Remove the common prefix to obtain shorter pathway labels in the heatmap.
hallmark_table$term <- sub("^HALLMARK_", "", hallmark_table$term)
hallmark_gene_sets <- split(hallmark_table$gene, hallmark_table$term)



# 5. Calculate Hallmark GSVA scores
# Gaussian kernel is retained from the original analysis.
gsva_parameters <- gsvaParam(
  grouped_expression,
  hallmark_gene_sets,
  kcdf = "Gaussian"
)
hallmark_scores <- gsva(gsva_parameters, verbose = FALSE)

write.csv(
  hallmark_scores,
  file.path(output_dir, "hallmark_gsva_scores.csv")
)


# 6. Select the pathways shown in the paper figure
pathway_order <- c(
  "ADIPOGENESIS",
  "ALLOGRAFT_REJECTION",
  "APICAL_SURFACE",
  "APOPTOSIS",
  "BILE_ACID_METABOLISM",
  "COAGULATION",
  "COMPLEMENT",
  "E2F_TARGETS",
  "EPITHELIAL_MESENCHYMAL_TRANSITION",
  "FATTY_ACID_METABOLISM",
  "G2M_CHECKPOINT",
  "HEDGEHOG_SIGNALING",
  "HYPOXIA",
  "IL2_STAT5_SIGNALING",
  "IL6_JAK_STAT3_SIGNALING",
  "INFLAMMATORY_RESPONSE",
  "INTERFERON_ALPHA_RESPONSE",
  "INTERFERON_GAMMA_RESPONSE",
  "KRAS_SIGNALING_DN",
  "KRAS_SIGNALING_UP",
  "MITOTIC_SPINDLE",
  "MTORC1_SIGNALING",
  "MYC_TARGETS_V1",
  "MYOGENESIS",
  "NOTCH_SIGNALING",
  "PEROXISOME",
  "PI3K_AKT_MTOR_SIGNALING",
  "PROTEIN_SECRETION",
  "REACTIVE_OXYGEN_SPECIES_PATHWAY",
  "TNFA_SIGNALING_VIA_NFKB",
  "UNFOLDED_PROTEIN_RESPONSE",
  "WNT_BETA_CATENIN_SIGNALING",
  "XENOBIOTIC_METABOLISM"
)

figure_scores <- hallmark_scores[pathway_order, , drop = FALSE]

write.csv(
  figure_scores,
  file.path(output_dir, "hallmark_gsva_scores_figure.csv")
)

# 7. Draw the Hallmark GSVA heatmap used in the paper

heatmap_colors <- colorRampPalette(
  rev(brewer.pal(n = 7, name = "RdBu"))
)(100)

pdf(
  file.path(output_dir, "Fig_Hallmark_GSVA_heatmap.pdf"),
  width = 8,
  height = 8
)

pheatmap(
  figure_scores,
  color = heatmap_colors,
  breaks = seq(-0.4, 0.4, length.out = 101),
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  show_rownames = TRUE,
  show_colnames = TRUE,
  fontsize = 10,
  fontsize_row = 12,
  fontsize_col = 10,
  angle_col = 45,
  border_color = "gray25"
)

dev.off()
