"""Optional training script for KG embeddings."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_triples_factory(path: Path) -> TriplesFactory:
    return TriplesFactory.from_path(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional KG embedding model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model artifacts.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    cfg = load_yaml(config_path)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg.get("training", {})
    optimizer_cfg = cfg.get("optimizer", {})
    evaluator_cfg = cfg.get("evaluator", {})

    train_path = Path(dataset_cfg["train_path"])
    valid_path = Path(dataset_cfg["valid_path"])
    test_path = Path(dataset_cfg["test_path"])

    print(f"[INFO] Loading train triples from: {train_path}")
    training = load_triples_factory(train_path)

    print(f"[INFO] Loading validation triples from: {valid_path}")
    validation = TriplesFactory.from_path(
        str(valid_path),
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    print(f"[INFO] Loading test triples from: {test_path}")
    testing = TriplesFactory.from_path(
        str(test_path),
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    print("[INFO] Starting training pipeline...")
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_cfg["name"],
        model_kwargs=model_cfg.get("kwargs", {}),
        training_kwargs=training_cfg,
        optimizer=optimizer_cfg.get("name", "Adam"),
        optimizer_kwargs=optimizer_cfg.get("kwargs", {}),
        evaluator=evaluator_cfg.get("name", "RankBasedEvaluator"),
        random_seed=cfg.get("random_seed", 42),
        device=cfg.get("device"),
    )

    print("[INFO] Saving pipeline artifacts...")
    result.save_to_directory(str(output_dir))

    metrics_path = output_dir / "training_metrics.json"
    metrics = result.metric_results.to_dict()
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    run_info = {
        "config_path": str(config_path),
        "model_name": model_cfg["name"],
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("[INFO] Training completed successfully.")
    print(f"[INFO] Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()