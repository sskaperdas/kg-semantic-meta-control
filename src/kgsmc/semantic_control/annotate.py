from __future__ import annotations

from copy import deepcopy

from kgsmc.semantic_control.checker import SemanticChecker


def annotate_record(
    record: dict,
    checker: SemanticChecker,
) -> dict:
    annotated = deepcopy(record)
    query = annotated["query"]

    for candidate in annotated["candidates"]:
        if annotated["query_type"] == "tail_prediction":
            result = checker.check_triple(
                head=query["head"],
                relation=query["relation"],
                tail=candidate["entity"],
            )
        else:
            result = checker.check_triple(
                head=candidate["entity"],
                relation=query["relation"],
                tail=query["tail"],
            )

        candidate["semantic"] = result.to_dict()

    return annotated