# SpaDiff

[![Documentation Status](https://readthedocs.org/projects/spadiff/badge/?version=latest)](https://spadiff.readthedocs.io/en/latest/)

Topology-aware score-based diffusion modeling for integrating multi-slice and multi-omics spatial data


## Abstract

Biological systems are organized across space and time, yet most spatial omics studies still rely on individual two-dimensional slices that provide only partial views of tissue architecture. As spatial profiling expands to serial slices, multiple conditions, and complementary molecular modalities, there is a growing need for computational methods that can integrate heterogeneous measurements while preserving spatial continuity and tissue topology.
Here we present SpaDiff, a spatial diffusion-dynamics framework for integrating, denoising, and generating multi-slice and multi-omics spatial data. 
SpaDiff represents tissue organization with simplicial complexes, enabling the modeling of higher-order spatial interactions beyond conventional edge-based graphs. It further formulates integration within a unified conditional score-based diffusion framework, in which diffusion processes defined on distinct simplicial complexes are coupled through a spatially constrained stochastic differential equation. 
This formulation enables harmonization across slices and modalities while preserving biologically meaningful spatial structure.
Across 19 spatial transcriptomics datasets from human and mouse tissues, SpaDiff improves cross-slice integration, maintains anatomical consistency, and recovers coherent spatial domains in both serial and non-serial settings. 
SpaDiff also generalizes to spatial multi-omics data, including joint analysis of spatial ATAC-RNA measurements. In HER2-positive breast cancer, SpaDiff supports crossmodal generation of gene expression from histology and identifies candidate prognostic genes.
Together, these results establish SpaDiff as a general framework for reconstructing tissue functional landscapes from complex spatial omics data.


## Method overview

The core workflow is:

```text
PCA/LSI features + spatial coordinates
    -> spatial graph and simplicial operators
    -> order-specific topology encoder and attention fusion
    -> conditional score-based diffusion
    -> integrated representations and denoised features
```

For each simplex order, SpaDiff propagates spot features over a normalized
node-simplex operator. The resulting order-specific embeddings are fused into
a topology representation `H`, which conditions the reverse diffusion process
together with slice, batch, or modality labels.

SpaDiff optimizes three manuscript-level objectives:

```text
L_total = lambda_DSM * L_DSM
        + lambda_align * L_align
        + lambda_KL * L_KL
```

`L_DSM` learns the conditional score, `L_align` reduces technical information
in the topology representation, and `L_KL` regularizes condition-specific
topology distributions toward a shared prior. In the implementation,
`batch_posterior_weight` controls an auxiliary posterior term inside
`L_align`; it is not a fourth main objective.

PCA/LSI preprocessing, coordinate alignment or external spatial weighting,
and downstream clustering are separate analytical steps surrounding the core
SpaDiff model.

## Runtime environment

| Component | Supported or reference version |
| --- | --- |
| Python | `>=3.9,<3.11` |
| PyTorch | `2.4.1` |
| CUDA | `12.4` reference build; optional |
| Scanpy / AnnData | `1.9.1` / `0.8.0` |
| NumPy / pandas | `1.24.1` / `1.4.2` |
| scikit-learn | `1.1.1` |
| R + mclust + rpy2 | Optional for mclust clustering |

A CUDA-capable GPU is recommended for the full tutorial training schedules.
CPU execution is supported but can be substantially slower.

## Installation

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/xy-geng/SpaDiff.git
cd SpaDiff

conda create -n spadiff python=3.9 -y
conda activate spadiff
python -m pip install --upgrade pip
```

### 2. Install PyTorch

For CUDA 12.4:

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only execution:

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Use the [official PyTorch installer](https://pytorch.org/get-started/previous-versions/)
if another CUDA build is required.

### 3. Install SpaDiff and the tutorial dependencies

```bash
python -m pip install -e ".[tutorial]"
```

The equivalent convenience command is:

```bash
python -m pip install -r requirements.txt
```

The core package uses native sparse PyTorch operators. PyTorch Geometric is
optional and can be installed with `python -m pip install -e ".[pyg]"`.

Verify the installation:

```bash
python -c "import SpaDiff as sd; print(sd.__version__)"
```

### 4. Optional: install mclust

The DLPFC tutorials use R `mclust` for downstream clustering. SpaDiff itself
does not require R; Louvain or Leiden can be used instead.

```bash
python -m pip install -e ".[mclust]"
R -e "install.packages('mclust', repos='https://cloud.r-project.org')"
```

## Input data

SpaDiff workflows use `AnnData`. At minimum:

- rows represent spots or pixels and `obs_names` are unique;
- columns represent genes for RNA or peaks for ATAC;
- `adata.obsm["spatial"]` stores finite spatial coordinates;
- multi-slice data include a categorical column such as
  `adata.obs["batch_name"]`;
- paired RNA and ATAC objects use matching spot names and coordinates.

RNA data are normalized and reduced to PCA features. Paired ATAC data can be
processed with `sd.robust_atac_lsi`. Multi-slice coordinates should be placed
in a common coordinate system before cross-slice topology construction. The
`align_serial_slices` utility implements the manuscript's global
Moran-centroid alignment.

For 10x Visium data, each sample directory should contain the filtered feature
matrix and the standard `spatial/` directory.
Reference labels such as`truth.txt` are used only for evaluation. 
Dataset sources are listed in [data/readme.md](data/readme.md).

## Quick start: multi-slice DLPFC integration

The following condensed example follows
[tutorial/1-DLPFC_multi_slice.ipynb](tutorial/1-DLPFC_multi_slice.ipynb).
Set `DATA_ROOT` before running it.

### 1. Load and preprocess the slices

```python
from pathlib import Path

import scanpy as sc
import torch

import SpaDiff as sd
from SpaDiff.spatial import spatial_reconstruction
from SpaDiff.utils import set_seed

SEED = 42
ST_SAMPLES = ["151673", "151674", "151675", "151676"]
BATCH_KEY = "batch_name"
DATA_ROOT = Path("/path/to/DLPFC")

set_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

slices = []
for sample in ST_SAMPLES:
    current = sc.read_visium(DATA_ROOT / sample)
    current.var_names_make_unique()
    current.layers["counts"] = current.X.copy()
    sc.pp.normalize_total(current, target_sum=1e4)
    sc.pp.log1p(current)
    current, _ = spatial_reconstruction(
        current, alpha=1.5, n_neighbors=10
    )
    current.obs[BATCH_KEY] = sample
    current.obs_names = [
        f"{sample}:{barcode}" for barcode in current.obs_names
    ]
    slices.append(current)

adata = sc.concat(slices, join="inner", merge="same")
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    layer="counts",
    n_top_genes=3000,
    batch_key=BATCH_KEY,
    subset=True,
)
```

### 2. Build multi-slice simplicial operators

```python
MAX_ORDER = 2
SIMPLEX_ORDERS = tuple(range(1, MAX_ORDER + 1))

topology = sd.build_spatial_topology(
    adata,
    mode="slice_aware",
    batch_key=BATCH_KEY,
    slice_order=ST_SAMPLES,
    k_intra=6,
    k_inter=2,
    max_order=MAX_ORDER,
    device=device,
)

sc.tl.pca(adata, n_comps=50)
features = torch.as_tensor(
    adata.obsm["X_pca"], dtype=torch.float32, device=device
)
```

`MAX_ORDER = 1` uses edges only, while `MAX_ORDER = 2` adds
triangle-induced interactions. Higher orders can increase memory use rapidly.

### 3. Train SpaDiff and obtain embeddings

```python
config = sd.SpaDiffConfig(
    data_dim=features.shape[1],
    condition_input_dim=features.shape[1],
    num_batches=len(ST_SAMPLES),
    simplex_orders=SIMPLEX_ORDERS,
)

model = sd.SpaDiff(config).to(device)
adata = model.fit_transform(
    adata,
    features,
    topology.operators,
    batch_key=BATCH_KEY,
    batch_order=ST_SAMPLES,
    reference_batch=ST_SAMPLES[0],
    epochs=500,
    progress=True,
)
```

Use `epochs=5` for a quick installation smoke test. Training diagnostics are
available in `model.training_result_`.

## Main parameters

### Data and topology construction

| Parameter | Typical value | Description |
| --- | ---: | --- |
| `mode` | `"slice_aware"` | Within-slice and consecutive-slice graph; use `"global_knn"` for one shared coordinate graph |
| `k_intra` / `k_inter` | `6` / `2` | Within- and cross-slice neighbors |
| `n_neighbors` | `10` | Neighbors for global kNN or single-slice reconstruction |
| `max_order` | `2` | Maximum simplex order to construct |
| `simplex_orders` | `(1, 2)` | Operator channels used by the topology encoder |

### `SpaDiffConfig`

| Parameter | Default | Description |
| --- | ---: | --- |
| `data_dim` | `50` | PCA/LSI target width |
| `condition_input_dim` | `50` | Feature width supplied to the topology encoder |
| `topology_dim` | `64` | Fused topology embedding width |
| `propagation_steps` | `5` | Higher-order propagation depth |
| `propagation_alpha` | `0.4` | Balance between local features and propagation |
| `num_batches` / `num_modalities` | `1` / `1` | Number of technical conditions |

### Loss and diffusion parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `dsm_weighting` | `"variance"` | DSM time weighting |
| `dsm_weight` | `1.0` | Denoising score-matching weight |
| `batch_alignment_weight` | `0.5` | Technical-alignment weight |
| `prior_kl_weight` | `1.0` | Shared-prior regularization weight |
| `beta_min` / `beta_max` | `0.1` / `20.0` | Linear VP-SDE noise schedule |

### Training and harmonization

| Parameter | Default | Description |
| --- | ---: | --- |
| `epochs` | `500` | Optimization steps used by `fit_transform` |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `ema_decay` | `0.990` | Exponential moving-average decay |
| `strength` | `0.10` | Forward-noise fraction used for harmonization |
| `ode_steps` | `300` | Probability-flow ODE steps |

Tutorials may override these values for a specific dataset. In particular,
single-condition examples can set technical-alignment terms to zero.

## Tutorials and documentation

| Tutorial | Workflow |
| --- | --- |
| [Single-slice DLPFC](tutorial/1-DLPFC.ipynb) | Spatial domains in one Visium section |
| [Multi-slice DLPFC](tutorial/1-DLPFC_multi_slice.ipynb) | Serial-section integration and reconstruction |
| [Mouse brain](tutorial/2-Mousebrain.ipynb) | Anterior/posterior slice stitching |
| [Breast cancer](tutorial/3-breastcancer.ipynb) | Multi-section integration and denoised expression |
| [Mouse brain ATAC-RNA](tutorial/4-MouseBrain_ATAC_RNA.ipynb) | Paired spatial multi-omics integration |

Additional resources:

- [Online documentation](https://spadiff.readthedocs.io/en/latest/)
- [Installation guide](docs/source/Installation/install.md)
- [Dataset sources](data/readme.md)
