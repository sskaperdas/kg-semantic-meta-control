from __future__ import annotations

import json
import gzip
from pathlib import Path
from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker
from kgsmc.semantic_control.candidate_io import iter_candidate_records
from kgsmc.semantic_control.annotate import annotate_record

class Annotator:
    def __init__(self, checker: SemanticChecker):
        self.checker=checker

    def file_processing(self,input_file: Path,output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        valid=0
        invalid=0
        unknown=0
        with gzip.open(output_file,"wt",encoding="utf-8") as f:
            for i in iter_candidate_records(input_file):
                annotate=annotate_record(i,self.checker)
                for j in annotate.get("candidates",[]):
                    s=str(j.get("semantic",{}).get("status","UNKNOWN")).upper()
                    if s=="VALID":
                        valid+=1
                    elif s=="INVALID":
                        invalid+=1
                    else:
                        unknown+=1
                f.write(json.dumps(annotate,ensure_ascii=False)+"\n")
        return{
            "VALID":valid,
            "INVALID":invalid,
            "UNKNOWN":unknown
        }

def main()->int:
    dataset_dir = Path("data/processed/drugbank_core")
    input_file = Path("03_TOPK_RESULTS/DRUGBANK/ROTATE/test_tail_top50.jsonl.gz")
    output_file = Path("runs/annotated_candidates/drugbank_rotate_test_tail_top50.jsonl.gz")
    resources = SemanticResources(dataset_dir)
    checker = SemanticChecker(resources)
    annotator = Annotator(checker)
    stats = annotator.file_processing(input_file, output_file)
    print(f"DrugBank Stats: {stats}\n")

    dataset_dir1 = Path("data/processed/eurostatkg_structural")
    input_file1 = Path("03_TOPK_RESULTS/EUROSTATKG/ROTATE/test_tail_top50.jsonl.gz")
    output_file1 = Path("runs/annotated_candidates/eurostat_rotate_test_tail_top50.jsonl.gz")
    resources1 = SemanticResources(dataset_dir1)
    checker1 = SemanticChecker(resources1)
    annotator1 = Annotator(checker1)
    stats1 = annotator1.file_processing(input_file1, output_file1)
    print(f"Eurostat Stats: {stats1}\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())