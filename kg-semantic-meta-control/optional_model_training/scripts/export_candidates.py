"""Export candidates from optional trained models."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pykeen.pipeline import PipelineResult
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory


def load_test_queries(
    test_path: Path,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> List[Tuple[str, str, str]]:
    df = pd.read_csv(test_path, sep="\t", header=None, names=["h", "r", "t"], dtype=str)

    queries = []
    for row in df.itertuples(index=False):
        if row.h in entity_to_id and row.r in relation_to_id and row.t in entity_to_id:
            queries.append((row.h, row.r, row.t))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-K candidate predictions from trained model.")
    parser.add_argument("--model-dir", required=True, help="Directory produced by train_embeddings.py")
    parser.add_argument("--test-path", required=True, help="Path to test.tsv")
    parser.add_argument("--output-path", required=True, help="Path to output JSONL file")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates per query")
    parser.add_argument(
        "--query-type",
        choices=["tail", "head"],
        default="tail",
        help="Whether to generate tail or head predictions",
    )
    parser.add_argument("--dataset-name", default="unknown_dataset")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trained pipeline from: {model_dir}")
    result = PipelineResult.from_directory(str(model_dir))
    model = result.model
    training_tf: TriplesFactory = result.training

    queries = load_test_queries(
        test_path=test_path,
        entity_to_id=training_tf.entity_to_id,
        relation_to_id=training_tf.relation_to_id,
    )

    print(f"[INFO] Loaded {len(queries)} queries from test set.")
    print(f"[INFO] Exporting top-{args.top_k} {args.query_type} predictions to {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for query_id, (h, r, t_gold) in enumerate(queries):
            if args.query_type == "tail":
                pred_df = predict_target(
                    model=model,
                    head=h,
                    relation=r,
                    triples_factory=training_tf,
                ).df
                pred_df = pred_df.head(args.top_k)

                for rank_idx, row in enumerate(pred_df.itertuples(index=False), start=1):
                    record = {
                        "dataset": args.dataset_name,
                        "split": "test",
                        "query_id": query_id,
                        "query_type": "tail",
                        "h": h,
                        "r": r,
                        "t": str(row.label),
                        "gold_t": t_gold,
                        "score_raw": float(row.score),
                        "rank_raw": rank_idx,
                        "k": args.top_k,
                        "model": model.__class__.__name__,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            else:
                pred_df = predict_target(
                    model=model,
                    relation=r,
                    tail=t_gold,
                    triples_factory=training_tf,
                ).df
                pred_df = pred_df.head(args.top_k)

                for rank_idx, row in enumerate(pred_df.itertuples(index=False), start=1):
                    record = {
                        "dataset": args.dataset_name,
                        "split": "test",
                        "query_id": query_id,
                        "query_type": "head",
                        "h": str(row.label),
                        "r": r,
                        "t": t_gold,
                        "gold_h": h,
                        "score_raw": float(row.score),
                        "rank_raw": rank_idx,
                        "k": args.top_k,
                        "model": model.__class__.__name__,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[INFO] Candidate export completed successfully.")


if __name__ == "__main__":
    main()