from __future__ import annotations

import argparse
import json

from kgsmc.semantic_control.candidate_io import iter_candidate_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    for index, record in enumerate(
        iter_candidate_records(args.file),
        start=1,
    ):
        print("=" * 80)
        print(json.dumps(record, indent=2, ensure_ascii=False))

        if index >= args.limit:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())