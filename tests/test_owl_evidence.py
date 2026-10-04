from pathlib import Path
from unittest.mock import MagicMock
from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker
from kgsmc.semantic_control.evidence import EvidenceExtractor


def test_owl_evidence():
    root=Path(__file__).resolve().parents[1]
    dataset_dir=root/"data"/"processed"/"drugbank_core"
    resources=SemanticResources(dataset_dir)
    checker=SemanticChecker(resources)
    extractor=EvidenceExtractor(resources,checker,graph_index=MagicMock())
    constraints=resources.constraint("has_target")
    domains=constraints.domain
    ranges=constraints.range
    valid_d=extractor._type_evidence("drug:DB00001",domains)
    valid_r=extractor._type_evidence("target:BE0000048",ranges)
    conflict=extractor._type_evidence("category:proteins",ranges)
    missing=extractor._type_evidence("drug:DB00000",domains)

    assert valid_d["status"]=="DIRECT_SUPPORT"
    assert valid_r["status"]=="DIRECT_SUPPORT"
    assert conflict["status"]=="DISJOINT_CONFLICT"
    assert missing["status"]=="MISSING_TYPE"
