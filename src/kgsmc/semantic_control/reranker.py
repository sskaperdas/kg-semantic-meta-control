from __future__ import annotations

from copy import deepcopy


PRIORITY = {
    "VALID": 0,
    "UNKNOWN": 1,
    "INVALID": 2,
}


def rerank_record(record: dict) -> dict:
    reranked = deepcopy(record)

    reranked["candidates"].sort(
        key=lambda candidate: (
            PRIORITY[candidate["semantic"]["status"]],
            candidate["rank"],
        )
    )

    for new_rank, candidate in enumerate(
        reranked["candidates"],
        start=1,
    ):
        candidate["original_rank"] = candidate["rank"]
        candidate["new_rank"] = new_rank

    valid_candidates = [
        candidate
        for candidate in reranked["candidates"]
        if candidate["semantic"]["status"] == "VALID"
    ]

    if valid_candidates:
        reranked["decision"] = {
            "action": "PREDICT",
            "selected_entity": valid_candidates[0]["entity"],
        }
    else:
        reranked["decision"] = {
            "action": "ABSTAIN",
            "selected_entity": None,
        }

    return reranked