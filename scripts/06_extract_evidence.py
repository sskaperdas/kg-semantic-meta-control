from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from kgsmc.semantic_control.candidate_io import (
    iter_candidate_records,
)
from kgsmc.semantic_control.checker import (
    SemanticChecker,
)
from kgsmc.semantic_control.evidence import (
    EvidenceExtractor,
    GraphEvidenceIndex,
    add_evidence_to_record,
)
from kgsmc.semantic_control.resources import (
    SemanticResources,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "data/processed/drugbank_core"
        ),
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(
            "data/processed/drugbank_core/train.tsv"
        ),
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "03_TOPK_RESULTS/"
            "DRUGBANK/"
            "ROTATE/"
            "valid_tail_top50.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "runs/evidence/"
            "drugbank_rotate_"
            "valid_tail_top50_evidence.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path(
            "runs/evidence/"
            "drugbank_rotate_"
            "valid_tail_evidence_summary.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    resources = SemanticResources(
        args.dataset_dir
    )

    checker = SemanticChecker(
        resources
    )

    print("Building training graph index...")

    graph_index = GraphEvidenceIndex(
        args.train_file
    )

    extractor = EvidenceExtractor(
        resources=resources,
        checker=checker,
        graph_index=graph_index,
    )

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_count = 0
    candidate_count = 0

    semantic_statuses = Counter()
    domain_statuses = Counter()
    range_statuses = Counter()
    graph_lengths = Counter()

    with gzip.open(
        args.output_file,
        "wt",
        encoding="utf-8",
    ) as writer:

        for record in tqdm(
            iter_candidate_records(
                args.input_file
            ),
            desc="Extracting evidence",
            unit="query",
        ):
            record = add_evidence_to_record(
                record,
                extractor,
            )

            query_count += 1

            for candidate in record["candidates"]:
                candidate_count += 1

                evidence = candidate["evidence"]

                semantic_statuses[
                    evidence["semantic_status"]
                ] += 1

                domain_statuses[
                    evidence["ontology"]
                    ["domain"]
                    ["status"]
                ] += 1

                range_statuses[
                    evidence["ontology"]
                    ["range"]
                    ["status"]
                ] += 1

                length = (
                    evidence["graph"]
                    ["shortest_witness_length"]
                )

                if length is None:
                    graph_lengths["NONE"] += 1
                else:
                    graph_lengths[str(length)] += 1

            writer.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                args.limit is not None
                and query_count >= args.limit
            ):
                break

    summary = {
        "queries": query_count,
        "candidates": candidate_count,
        "semantic_statuses": dict(
            semantic_statuses
        ),
        "domain_evidence": dict(
            domain_statuses
        ),
        "range_evidence": dict(
            range_statuses
        ),
        "graph_witness_length": dict(
            graph_lengths
        ),
    }

    args.summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"Evidence file: {args.output_file}"
    )

    print(
        f"Summary: {args.summary_file}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())