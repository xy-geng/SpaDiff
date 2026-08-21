# Install SpaDiff

## Supported environment

The reference SpaDiff environment uses the following versions. Python 3.9 is
recommended because it matches the tutorial environment; Python 3.10 is also
supported.

| Component | Supported or reference version | Purpose |
| --- | --- | --- |
| Python | `>=3.9,<3.11` | Runtime |
| PyTorch | `2.4.1` | Diffusion model and sparse operators |
| CUDA | `12.4` reference build | Optional GPU acceleration |
| NumPy | `1.24.1` | Numerical arrays |
| pandas | `1.4.2` | Observation and feature metadata |
| SciPy | `>=1.9,<1.11` | Sparse matrices and scientific routines |
| scikit-learn | `1.1.1` | PCA, LSI, and nearest neighbors |
| Scanpy | `1.9.1` | Preprocessing and visualization |
| AnnData | `0.8.0` | Spatial-omics data container |
| PyTorch Geometric | `2.6.1` | Optional graph workflows |
| R, `mclust`, and `rpy2` | `rpy2==3.4.1` | Optional mclust clustering |

An NVIDIA GPU is strongly recommended for the full 300–500 epoch examples.
CPU execution is supported, but training and probability-flow sampling can be
substantially slower. Memory usage depends on both the number of spots and the
number of higher-order simplices; begin with edges only (`MAX_ORDER = 1`) if a
dense graph exhausts memory.

## 1. Clone the repository and create an environment

```bash
git clone https://github.com/xy-geng/SpaDiff.git
cd SpaDiff

conda create -n spadiff python=3.9 -y
conda activate spadiff
python -m pip install --upgrade pip
```

Run all remaining commands from the repository root.

## 2. Install PyTorch

Choose **one** build. For an NVIDIA GPU compatible with CUDA 12.4:

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only execution:

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

If you need another CUDA build, select the matching command from the
[official PyTorch installer](https://pytorch.org/get-started/previous-versions/).
The PyTorch package, CUDA wheel, and local NVIDIA driver must be compatible.

## 3. Install SpaDiff and the tutorial stack

Install the repository in editable mode so that local source changes are
available immediately:

```bash
python -m pip install -e ".[tutorial]"
```

The equivalent convenience command is:

```bash
python -m pip install -r requirements.txt
```

The core implementation uses native sparse PyTorch operators. PyTorch
Geometric is optional and only needed for additional PyG workflows:

```bash
python -m pip install -e ".[pyg]"
```

When compiled PyG extensions are required, use wheels that match your PyTorch
and CUDA versions. Consult the
[PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html).

## 4. Optional mclust support

The reference DLPFC evaluation uses the R package `mclust`. The SpaDiff model
itself does not require R, and you can use Leiden or Louvain clustering from
Scanpy instead.

```bash
python -m pip install -e ".[mclust]"
R -e "install.packages('mclust', repos='https://cloud.r-project.org')"
```

Make sure `R` is on `PATH`. On some systems, `rpy2` also requires `R_HOME` to
point to the active R installation.

## 5. Verify the installation

```bash
python -c "import SpaDiff as sd; print('SpaDiff', sd.__version__)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

Launch the tutorials with:

```bash
jupyter lab docs/source/tutorial
```

The documentation notebooks do not execute during a Read the Docs build. Run
them locally after changing each `DATA_ROOT` to the location of your data.

## Input data conventions

All workflows use `AnnData`. At minimum:

- rows are spots or pixels and `obs_names` are unique;
- columns are genes (RNA) or peaks (ATAC);
- `adata.X` contains non-negative raw counts before preprocessing;
- `adata.obsm["spatial"]` contains an `n_spots × 2` coordinate matrix;
- multi-slice inputs use a categorical observation column such as
  `adata.obs["batch_name"]`;
- paired RNA and ATAC objects contain the same spot names and coordinates.

For 10x Visium samples, each sample directory should contain the filtered
feature matrix and the standard `spatial/` directory. A `truth.txt` file is
used only for tutorial evaluation and may be omitted when reference labels are
unavailable.

## Common installation problems

**`torch.cuda.is_available()` is `False`**
: Confirm that you installed a CUDA-enabled PyTorch wheel and that the NVIDIA
  driver supports it. SpaDiff will otherwise fall back to CPU.

**PyTorch Geometric reports an undefined symbol or missing library**
: Reinstall its compiled extensions using wheels built for the exact installed
  PyTorch and CUDA versions, or omit the optional PyG extra.

**`rpy2` cannot find R**
: Verify `R --version`, set `R_HOME` if required, and restart the shell before
  reinstalling `rpy2`.

**Simplicial construction uses too much memory**
: Reduce `N_NEIGHBORS`, use `MAX_ORDER = 1`, or test on a spatial subset before
  enabling triangles (`MAX_ORDER = 2`).
