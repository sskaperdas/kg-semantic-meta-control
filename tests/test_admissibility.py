from pathlib import Path
from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker

def test_drugbank_admissibility():
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "data" / "processed" / "drugbank_core"
    resources = SemanticResources(dataset_dir)
    checker = SemanticChecker(resources)

    valid_result = checker.check_triple(
        head="drug:DB00001",
        relation="has_target",
        tail="target:BE0000048",
    )

    invalid_result = checker.check_triple(
        head="drug:DB00001",
        relation="has_target",
        tail="category:proteins",
    )

    unknown_result = checker.check_triple(
        head="drug:DB00001",
        relation="has_target1",
        tail="target:BE0000048",
    )

    unknown_result1 = checker.check_triple(
        head="drug:DB00000",
        relation="has_target",
        tail="target:BE0000048",
    )

    assert valid_result.status == "VALID"
    assert invalid_result.status == "INVALID"
    assert unknown_result.status == "UNKNOWN"
    assert unknown_result1.status == "UNKNOWN"

def test_eurostat_admissibility():
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "data" / "processed" / "eurostatkg_structural"
    resources = SemanticResources(dataset_dir)
    checker = SemanticChecker(resources)

    valid_result = checker.check_triple(
        head="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/referenceData934",
        relation="https://ec.europa.eu/eurostat/NLP4StatRef/ontology/hasURI",
        tail="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/SEArticle613",
    )

    invalid_result = checker.check_triple(
        head="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/referenceData934",
        relation="https://ec.europa.eu/eurostat/NLP4StatRef/ontology/hasURI",
        tail="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/agricultur",
    )

    unknown_result = checker.check_triple(
        head="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/referenceData934",
        relation="https://ec.europa.eu/eurostat/NLP4StatRef/ontology/hasURIs",
        tail="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/SEArticle613",
    )

    unknown_result1 = checker.check_triple(
        head="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/SEArticle623",
        relation="https://ec.europa.eu/eurostat/NLP4StatRef/ontology/hasURI",
        tail="https://ec.europa.eu/eurostat/NLP4StatRef/knowledge/SEArticle613",
    )

    assert valid_result.status == "VALID"
    assert invalid_result.status == "INVALID"
    assert unknown_result.status == "UNKNOWN"
    assert unknown_result1.status == "UNKNOWN"
