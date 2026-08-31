from __future__ import annotations

import argparse
import gzip
import json
import numpy as np
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

def statistics(stat: list[int|float]) -> dict[str,float]:
    if not stat:
        return {"mean":0.0,"median":0.0,"p90":0.0,"max":0.0}
    array=np.array(stat)
    return {
        "mean": round(float(np.mean(array)), 4),
        "median":round(float(np.median(array)),4),
        "p90":round(float(np.percentile(array,90)),4),
        "max":round(float(np.max(array)),4),

    }

def dataset_processing(dataset:str, model:str="ROTATE", limit:int| None=None)-> None:
    if dataset=="drugbank":
        folder="drugbank_core"
        topk_folder="DRUGBANK"
    elif dataset=="eurostat":
        folder="eurostatkg_structural"
        topk_folder="EUROSTATKG"
    else:
        raise ValueError("Unknown dataset")

    dataset_dir= Path(f"data/processed/{folder}")
    train_file= Path(f"data/processed/{folder}/train.tsv")
    input_file= Path(f"03_TOPK_RESULTS/{topk_folder}/{model}/valid_tail_top50.jsonl.gz")
    output_file= Path(f"runs/evidence/{dataset}_{model.lower()}_valid_tail_top50_evidence_v02.jsonl.gz")
    summary_file= Path(f"runs/evidence/{dataset}_{model.lower()}_valid_tail_evidence_summary_v02.json")

    if not input_file.exists():
        print("Input file not found")
        return

    resources = SemanticResources(
        dataset_dir
    )

    checker = SemanticChecker(
        resources
    )

    print("Building training graph index...")

    graph_index = GraphEvidenceIndex(
        train_file
    )

    extractor = EvidenceExtractor(
        resources=resources,
        checker=checker,
        graph_index=graph_index,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_count = 0
    candidate_count = 0

    semantic_statuses = Counter()
    domain_statuses = Counter()
    range_statuses = Counter()
    graph_lengths = Counter()
    relation_role_statuses=Counter()
    three_hop_counts:list[int]=[]
    relation_role_counts:list[int]=[]

    with gzip.open(
            output_file,
            "wt",
            encoding="utf-8",
    ) as writer:

        for record in tqdm(
                iter_candidate_records(
                    input_file
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
                if length==1:
                    graph_lengths["1"]+=1
                elif length==2:
                    graph_lengths["2"]+=1
                elif evidence["graph"].get("has_three_hop_witness",False):
                    graph_lengths["3"]+=1
                else:
                    graph_lengths["NONE"] += 1


                role_supported = evidence["graph"].get("relation_role_supported",False)
                relation_role_statuses[str(role_supported)]+=1

                three_hop_counts.append(
                    evidence["graph"].get("three_hop_count",0)
                )
                relation_role_counts.append(
                    evidence["graph"].get("relation_role_count",0)
                )

            writer.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                    limit is not None
                    and query_count >= limit
            ):
                break

    summary = {
        "dataset": dataset,
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
        "graph":{
            "1-hop count": graph_lengths.get("1",0),
            "2-hop count": graph_lengths.get("2",0),
            "3-hop count": graph_lengths.get("3",0),
            "no-witness": graph_lengths.get("NONE",0),
        },
        "three_hop_witness_distribution": statistics(
            three_hop_counts
        ),
        "relation_role": {
            "supported": relation_role_statuses.get("True",0),
            "unsupported": relation_role_statuses.get("False",0),
            "relation_role_count_distribution": statistics(
                relation_role_counts
            ),
        },
    }

    summary_file.write_text(
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
        f"Evidence file: {output_file}"
    )

    print(
        f"Summary: {summary_file}"
    )

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["drugbank","eurostat"],
    )

    parser.add_argument(
        "--model",
        type=str,
        default= "ROTATE",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args=parser.parse_args()
    for i in args.datasets:
        dataset_processing(dataset=i, model=args.model, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())