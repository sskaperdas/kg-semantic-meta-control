from __future__ import annotations
from pathlib import Path
import pytest

from kgsmc.semantic_control.evidence import GraphEvidenceIndex

@pytest.fixture
def train_file(tmp_path: Path)->Path:
    path = tmp_path / "train.tsv"
    triples=[
        "A\tr1\tB",
        "B\tr2\tC",
        "C\tr3\tD",
        "A\tr4\tF",
        "A\tr5\tE",
        "E\tr6\tA",
        "A\tr7\tD",
    ]
    path.write_text("\n".join(triples),encoding="utf-8")
    return path

def test_positive(train_file: Path)-> None:
    index=GraphEvidenceIndex(train_file)
    count,examples=index.three_hop_witness("A")
    assert count["D"]==1
    assert examples["D"]==["A","r1","B","r2","C","r3","D"]

def test_negative(train_file: Path)-> None:
    index=GraphEvidenceIndex(train_file)
    count,examples=index.three_hop_witness("A")
    assert count.get("F",0)==0
    assert "F" not in examples

def test_non_backtracking(train_file: Path)-> None:
    index=GraphEvidenceIndex(train_file)
    count,examples=index.three_hop_witness("A")
    assert count["D"]==1