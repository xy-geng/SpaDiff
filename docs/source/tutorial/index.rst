Tutorials
=========

The tutorials cover four core workflows and one gene-denoising example.

.. list-table:: Workflow guide
   :header-rows: 1
   :widths: 24 28

   * - Workflow
     - Dataset
   * - Single-slice spatial domain identification
     - Human DLPFC section 151674
   * - Three-dimensional reconstruction from serial sections
     - Four consecutive human DLPFC sections
   * - Adjacent-slice alignment and stitching
     - Anterior and posterior mouse sagittal brain sections
   * - Spatial multi-omics integration
     - P21/P22 mouse coronal brain spatial ATAC-RNA-seq
   * - Gene-expression denoising
     - Three HER2-positive breast-cancer sections

Before running a notebook, complete :doc:`../Installation/install` and replace
its ``DATA_ROOT`` placeholder. The reference training schedules reproduce the
intended experiment; use 5 epochs for a quick installation smoke test.

Core workflows
--------------

.. toctree::
   :maxdepth: 1

   Single-slice spatial domain identification <01_DLPFC/1-DLPFC>
   Three-dimensional reconstruction from serial sections <01_DLPFC/1-DLPFC_multi_slice>
   Adjacent-slice alignment and stitching <02_MouseBrain/2-Mousebrain>
   Paired spatial ATAC-RNA integration <04_MultiOmics/4-MouseBrain_ATAC_RNA>

Additional example
------------------

The breast-cancer example also maps harmonized PCA features back to gene
space and stores a denoised expression layer.

.. toctree::
   :maxdepth: 1

   Breast-cancer integration and gene denoising <03_BreastCancer/3-breastcancer>

.. toctree::
   :hidden:

   01_DLPFC/index
   02_MouseBrain/index
   03_BreastCancer/index
   04_MultiOmics/index
