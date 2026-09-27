from pathlib import Path

from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker


def test_checker_first_100():
    root = Path(__file__).resolve().parents[1]

    dataset_dir = root / "data" / "processed" / "drugbank_core"
    test_file = dataset_dir / "test.tsv"

    resources = SemanticResources(dataset_dir)
    checker = SemanticChecker(resources)

    checked = 0

    with test_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            head, relation, tail = line.split("\t")

            result = checker.check_triple(
                head,
                relation,
                tail,
            )

            assert result.status == "VALID"

            checked += 1

            if checked == 100:
                break

    assert checked == 100
