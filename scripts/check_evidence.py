from __future__ import annotations

import argparse
import json
from pathlib import Path
from kgsmc.semantic_control.candidate_io import iter_candidate_records

def check_evidence(path:Path,records: int=5, candidates:int=5):
    for count,record in enumerate(iter_candidate_records(path),start=1):
        if count>records:
            break

        if candidates is not None:
            record["candidates"]=record["candidates"][:candidates]
        print(f"\n Record #{count}")
        print(json.dumps(record,indent=2,ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["drugbank", "eurostat"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ROTATE",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,)
    args = parser.parse_args()
    for i in args.datasets:
        path=Path(
            f"runs/evidence/{i}_{args.model.lower()}_valid_tail_top50_evidence_v02.jsonl.gz"
        )
        check_evidence(path, args.records, args.candidates)


    return 0

if __name__ == "__main__":
    raise SystemExit(main())