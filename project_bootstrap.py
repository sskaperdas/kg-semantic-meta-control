from pathlib import Path

PROJECT_NAME = "kg-semantic-meta-control"

DIRS = [
    "configs/datasets",
    "configs/pipeline",
    "configs/training_optional",

    "data/raw/eurostatkg/source",
    "data/raw/eurostatkg/ontology",
    "data/raw/eurostatkg/graph",
    "data/raw/bio2rdf/source",
    "data/raw/bio2rdf/ontology",
    "data/raw/bio2rdf/graph",

    "data/interim/eurostatkg",
    "data/interim/bio2rdf",

    "data/processed/eurostatkg",
    "data/processed/bio2rdf",

    "data/external",

    "indexes/ontology",
    "indexes/graph",

    "runs/baseline",
    "runs/pipeline",
    "runs/evaluation",
    "runs/logs",

    "outputs/predictions",
    "outputs/traces",
    "outputs/reports",
    "outputs/figures",

    "src/kgsmc/utils",
    "src/kgsmc/data",
    "src/kgsmc/ontology",
    "src/kgsmc/graph",
    "src/kgsmc/predictors",
    "src/kgsmc/semantic_checks",
    "src/kgsmc/evidence",
    "src/kgsmc/gating",
    "src/kgsmc/evaluation",
    "src/kgsmc/pipelines",

    "scripts",
    "scripts/dataset_builders",

    "notebooks",
    "tests",

    "optional_model_training/configs",
    "optional_model_training/scripts",
    "optional_model_training/checkpoints",
    "optional_model_training/runs",
    "optional_model_training/notes",
]

FILES = {
    "README.md": """# KG Semantic Meta-Control

This project implements a post-hoc semantic meta-control pipeline for Knowledge Graph link prediction.

## Main objective
The core objective is **not** model training, but post-hoc semantic validation, evidence extraction, gating, and explainable ranking.

## Optional area
The folder `optional_model_training/` contains separate utilities for training embedding models if needed for experimentation.
""",
    ".gitignore": """__pycache__/
*.pyc
.venv/
env/
.ipynb_checkpoints/
data/raw/
data/external/
indexes/
runs/
outputs/
optional_model_training/checkpoints/
optional_model_training/runs/
""",
    "requirements.txt": """pandas
pyyaml
tqdm
rdflib
owlready2
networkx
pykeen
pytest
""",
    "configs/project.yaml": """project_name: kg-semantic-meta-control
default_dataset: eurostatkg
default_model: TransE
top_k: 50
""",
    "configs/datasets/eurostatkg.yaml": """name: eurostatkg
raw_path: data/raw/eurostatkg
processed_path: data/processed/dataset1
ontology_path: data/raw/eurostatkg/ontology.owl
train_path: data/processed/dataset1/train.tsv
valid_path: data/processed/dataset1/valid.tsv
test_path: data/processed/dataset1/test.tsv
""",
    "configs/datasets/bio2rdf.yaml": """name: bio2rdf
raw_path: data/raw/bio2rdf
processed_path: data/processed/dataset2
ontology_path: data/raw/bio2rdf/ontology.owl
train_path: data/processed/dataset2/train.tsv
valid_path: data/processed/dataset2/valid.tsv
test_path: data/processed/dataset2/test.tsv
""",
    "configs/pipeline/baseline.yaml": """model_name: TransE
top_k: 50
query_types:
  - tail
""",
    "configs/pipeline/gating.yaml": """alpha: 0.5
tau: 0.4
epsilon: 0.0
lambda_score: 0.5
mu_penalty: 0.25
""",
    "configs/pipeline/evaluation.yaml": """metrics:
  - mrr
  - hits_at_10
  - svr
  - valid_at_k
  - coverage
  - explainability_coverage
""",
    "configs/training_optional/transe.yaml": """model: TransE
epochs: 50
embedding_dim: 100
batch_size: 256
""",
    "configs/training_optional/complex.yaml": """model: ComplEx
epochs: 50
embedding_dim: 100
batch_size: 256
""",
    "src/kgsmc/__init__.py": "",
    "src/kgsmc/constants.py": '''"""Project-wide constants."""\n''',
    "src/kgsmc/utils/__init__.py": "",
    "src/kgsmc/utils/io_utils.py": '''"""I/O helpers."""\n''',
    "src/kgsmc/utils/logging_utils.py": '''"""Logging helpers."""\n''',
    "src/kgsmc/utils/config_utils.py": '''"""Config loading helpers."""\n''',
    "src/kgsmc/utils/path_utils.py": '''"""Path helpers."""\n''',
    "src/kgsmc/data/__init__.py": "",
    "src/kgsmc/data/loaders.py": '''"""Dataset loading utilities."""\n''',
    "src/kgsmc/data/preprocess.py": '''"""Dataset preprocessing utilities."""\n''',
    "src/kgsmc/data/splitters.py": '''"""Train/valid/test split helpers."""\n''',
    "src/kgsmc/data/schemas.py": '''"""Data schemas for triples and predictions."""\n''',
    "src/kgsmc/ontology/__init__.py": "",
    "src/kgsmc/ontology/ontology_loader.py": '''"""Ontology loading functions."""\n''',
    "src/kgsmc/ontology/constraint_extractor.py": '''"""Extract domain/range/disjointness constraints."""\n''',
    "src/kgsmc/ontology/subclass_index.py": '''"""Subclass closure utilities."""\n''',
    "src/kgsmc/ontology/type_index.py": '''"""Entity type index utilities."""\n''',
    "src/kgsmc/graph/__init__.py": "",
    "src/kgsmc/graph/graph_index.py": '''"""Graph indexing utilities."""\n''',
    "src/kgsmc/graph/adjacency.py": '''"""Adjacency construction helpers."""\n''',
    "src/kgsmc/graph/path_search.py": '''"""Bounded path search utilities."""\n''',
    "src/kgsmc/predictors/__init__.py": "",
    "src/kgsmc/predictors/base.py": '''"""Base predictor interface."""\n''',
    "src/kgsmc/predictors/pykeen_adapter.py": '''"""PyKEEN-based predictor adapter."""\n''',
    "src/kgsmc/predictors/mock_predictor.py": '''"""Mock predictor for dry-runs/tests."""\n''',
    "src/kgsmc/semantic_checks/__init__.py": "",
    "src/kgsmc/semantic_checks/admissibility.py": '''"""Main admissibility checks."""\n''',
    "src/kgsmc/semantic_checks/domain_range.py": '''"""Domain/range checks."""\n''',
    "src/kgsmc/semantic_checks/disjointness.py": '''"""Disjointness checks."""\n''',
    "src/kgsmc/evidence/__init__.py": "",
    "src/kgsmc/evidence/owl_evidence.py": '''"""Ontology-based evidence scoring."""\n''',
    "src/kgsmc/evidence/graph_evidence.py": '''"""Graph/path-based evidence scoring."""\n''',
    "src/kgsmc/evidence/aggregation.py": '''"""Evidence aggregation logic."""\n''',
    "src/kgsmc/evidence/traces.py": '''"""Evidence trace builders."""\n''',
    "src/kgsmc/gating/__init__.py": "",
    "src/kgsmc/gating/policy.py": '''"""Accept/demote/abstain policy."""\n''',
    "src/kgsmc/gating/scoring.py": '''"""Final score computation."""\n''',
    "src/kgsmc/gating/ranker.py": '''"""Final ranking logic."""\n''',
    "src/kgsmc/evaluation/__init__.py": "",
    "src/kgsmc/evaluation/ranking_metrics.py": '''"""MRR, Hits@K, etc."""\n''',
    "src/kgsmc/evaluation/semantic_metrics.py": '''"""SVR, Valid@K, etc."""\n''',
    "src/kgsmc/evaluation/coverage_metrics.py": '''"""Coverage and abstention metrics."""\n''',
    "src/kgsmc/evaluation/report_builder.py": '''"""Evaluation report generation."""\n''',
    "src/kgsmc/pipelines/__init__.py": "",
    "src/kgsmc/pipelines/run_baseline.py": '''"""Run baseline prediction export."""\n''',
    "src/kgsmc/pipelines/run_indexing.py": '''"""Build graph and ontology indexes."""\n''',
    "src/kgsmc/pipelines/run_meta_control.py": '''"""Run evidence-gated semantic meta-control."""\n''',
    "src/kgsmc/pipelines/run_evaluation.py": '''"""Run evaluation suite."""\n''',
    "scripts/prepare_data.py": '''"""CLI script for dataset preparation."""\n''',
    "scripts/build_indexes.py": '''"""CLI script for ontology/graph indexing."""\n''',
    "scripts/export_baseline_predictions.py": '''"""CLI script to export top-K baseline predictions."""\n''',
    "scripts/run_pipeline.py": '''"""CLI script to run the full main pipeline."""\n''',
    "scripts/evaluate_pipeline.py": '''"""CLI script to evaluate pipeline outputs."""\n''',
    "notebooks/01_data_inspection.ipynb": "",
    "notebooks/02_ontology_checks.ipynb": "",
    "notebooks/03_error_analysis.ipynb": "",
    "notebooks/04_results_plots.ipynb": "",
    "tests/__init__.py": "",
    "tests/test_admissibility.py": '''def test_placeholder():\n    assert True\n''',
    "tests/test_owl_evidence.py": '''def test_placeholder():\n    assert True\n''',
    "tests/test_graph_evidence.py": '''def test_placeholder():\n    assert True\n''',
    "tests/test_gating_policy.py": '''def test_placeholder():\n    assert True\n''',
    "optional_model_training/README.md": """# Optional Model Training

This folder is intentionally separated from the main pipeline.

Its purpose is to allow the student to:
- train at least one embedding model once,
- understand how candidate predictions and raw scores are produced,
- compare externally supplied predictions with self-trained baselines.

This is **recommended**, but it is **not the main research object** of the thesis.
""",
    "optional_model_training/configs/train_transe.yaml": """model: TransE
epochs: 50
embedding_dim: 100
batch_size: 256
""",
    "optional_model_training/configs/train_complex.yaml": """model: ComplEx
epochs: 50
embedding_dim: 100
batch_size: 256
""",
    "optional_model_training/scripts/train_embeddings.py": '''"""Optional training script for KG embeddings."""\n''',
    "optional_model_training/scripts/evaluate_embeddings.py": '''"""Optional embedding evaluation script."""\n''',
    "optional_model_training/scripts/export_candidates.py": '''"""Export candidates from optional trained models."""\n''',
}


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(PROJECT_NAME)
    root.mkdir(parents=True, exist_ok=True)

    for directory in DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for rel_path, content in FILES.items():
        write_file(root / rel_path, content)

    print(f"Project bootstrap created at: {root.resolve()}")


if __name__ == "__main__":
    main()