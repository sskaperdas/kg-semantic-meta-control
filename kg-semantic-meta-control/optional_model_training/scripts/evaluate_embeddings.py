"""Optional embedding evaluation script."""
import argparse
import json
from pathlib import Path

from pykeen.pipeline import PipelineResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained optional KG embedding model.")
    parser.add_argument("--model-dir", required=True, help="Directory produced by train_embeddings.py")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print(f"[INFO] Loading trained pipeline from: {model_dir}")
    result = PipelineResult.from_directory(str(model_dir))

    metrics = result.metric_results.to_dict()

    print("[INFO] Evaluation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_path = model_dir / "evaluation_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()