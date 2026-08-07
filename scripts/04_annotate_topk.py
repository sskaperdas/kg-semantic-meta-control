from __future__ import annotations

import argparse
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
        # valid=0
        # invalid=0
        # unknown=0
        with gzip.open(output_file,"wt",encoding="utf-8") as f:
            for i in iter_candidate_records(input_file):
                annotate=annotate_record(i,self.checker)
                # for j in annotate.get("candidates",[]):
                #     s=str(j.get("semantic",{}).get("status","UNKNOWN")).upper()
                #     if s=="VALID":
                #         valid+=1
                #     elif s=="INVALID":
                #         invalid+=1
                #     else:
                #         unknown+=1
                f.write(json.dumps(annotate,ensure_ascii=False)+"\n")
        # return{
        #     "VALID":valid,
        #     "INVALID":invalid,
        #     "UNKNOWN":unknown
        # }

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",type=Path,default=Path("data/processed/drugbank_core"))
    parser.add_argument("--input-file",type=Path,default=Path("03_TOPK_RESULTS/DRUGBANK/ROTATE/test_tail_top50.jsonl.gz"))
    parser.add_argument("--output-file",type=Path,default=Path("runs/annotated_candidates/drugbank_rotate_test_tail_top50.jsonl.gz"))
    args=parser.parse_args()
    resources=SemanticResources(args.dataset_dir)
    checker=SemanticChecker(resources)
    annotator=Annotator(checker)
    annotator.file_processing(args.input_file,args.output_file)
    # print(f"Stats: {stats}\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())