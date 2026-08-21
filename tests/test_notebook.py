from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    ROOT / "tutorial" / "1-DLPFC.ipynb": (
        ROOT / "docs" / "source" / "SpaDiff" / "01_DLPFC" / "1-DLPFC.ipynb"
    ),
    ROOT / "tutorial" / "1-DLPFC_multi_slice.ipynb": (
        ROOT
        / "docs"
        / "source"
        / "SpaDiff"
        / "01_DLPFC"
        / "1-DLPFC_multi_slice.ipynb"
    ),
    ROOT / "tutorial" / "2-Mousebrain.ipynb": (
        ROOT
        / "docs"
        / "source"
        / "SpaDiff"
        / "02_MouseBrain"
        / "2-Mousebrain.ipynb"
    ),
    ROOT / "tutorial" / "3-breastcancer.ipynb": (
        ROOT
        / "docs"
        / "source"
        / "SpaDiff"
        / "03_BreastCancer"
        / "3-breastcancer.ipynb"
    ),
}
MULTIOMICS_NOTEBOOK = ROOT / "tutorial" / "4-MouseBrain_ATAC_RNA.ipynb"
MULTIOMICS_DOCUMENTATION_COPY = (
    ROOT
    / "docs"
    / "source"
    / "tutorial"
    / "04_MultiOmics"
    / "4-MouseBrain_ATAC_RNA.ipynb"
)


def test_notebook_python_cells_parse_and_use_high_level_api():
    for tutorial in NOTEBOOKS:
        notebook = json.loads(tutorial.read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(source)
        assert "model.fit_transform(" in source
        assert "SpaDiffPipeline" not in source
        assert "train_spadiff" not in source
        assert "batch_ids =" not in source
        assert "modality_ids =" not in source
        assert "verbose_every" not in source


def test_documentation_notebook_is_synchronized():
    for tutorial, documentation_copy in NOTEBOOKS.items():
        assert documentation_copy.read_bytes() == tutorial.read_bytes()
    assert (
        MULTIOMICS_DOCUMENTATION_COPY.read_bytes()
        == MULTIOMICS_NOTEBOOK.read_bytes()
    )


def test_multiomics_notebook_follows_paper_fusion():
    notebook = json.loads(MULTIOMICS_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(source)

    assert "N_COMPONENTS = 50" in source
    assert "TRAINING_EPOCHS = 500" in source
    assert "sc.pp.scale(rna_model" in source
    assert "robust_atac_lsi" in source
    assert "mean_paired_embeddings" in source
    assert 'obsm["spadiff"] = joint_topology_np' in source
    assert 'used_obsm="spadiff"' in source

    assert "condition_features = torch.cat((atac_features" not in source
    assert "paired_alignment_loss" not in source
    assert "consensus_disagreement_embedding" not in source
    assert "build_adaptive_multimodal_connectivity" not in source
    assert "KMeans" not in source
