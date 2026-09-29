from __future__ import annotations

import argparse
import json
from pathlib import Path
from kgsmc.semantic_control.candidate_io import iter_candidate_records
from kgsmc.semantic_control.reranker import rerank_record

class Evaluator:
    def __init__(self):
        self.total=0
        self.mrr=0
        self.hits1=0
        self.hits10=0
        self.valid1=0
        self.valid10=0
        self.rr_mrr=0
        self.rr_hits1=0
        self.rr_hits10=0
        self.rr_valid1=0
        self.rr_valid10=0
        self.predict=0
        self.abstain=0

    def process(self,record:dict)->None:
        self.total+=1
        baseline_candidates=record.get("candidates",[])
        rerank=rerank_record(record)
        reranked_candidates=rerank.get("candidates",[])
        self.baseline_evaluate(baseline_candidates)
        self.reranked_evaluate(reranked_candidates)
        decision=rerank.get("decision",{}).get("action")
        if decision=="PREDICT":
            self.predict+=1
        elif decision=="ABSTAIN":
            self.abstain+=1

    def baseline_evaluate(self,candidates:list)->None:
        for i in candidates:
            if i.get("is_gold"):
                rank = i["rank"]
                self.mrr += 1.0 / rank
                if rank == 1:
                    self.hits1 += 1
                if rank <= 10:
                    self.hits10 += 1
                break
        if candidates:
            first=candidates[0]
            s=first.get("semantic",{}).get("status")
            if s=="VALID":
                self.valid1+=1
        valid=0
        count=0
        for i in candidates:
            if count==10:
                break
            s=i.get("semantic",{}).get("status")
            if s == "VALID":
                valid+=1
            count+=1
        if count>0:
            self.valid10+=valid/count

    def reranked_evaluate(self,candidates:list)->None:
        for i in candidates:
            if i.get("is_gold"):
                new=i["new_rank"]
                self.rr_mrr+=1.0/new
                if new==1:
                    self.rr_hits1+=1
                if new<=10:
                    self.rr_hits10+=1
                break
        if candidates:
            first=candidates[0]
            s=first.get("semantic",{}).get("status")
            if s=="VALID":
                self.rr_valid1+=1
        valid = 0
        count = 0
        for i in candidates:
            if count == 10:
                break
            s = i.get("semantic", {}).get("status")
            if s == "VALID":
                valid += 1
            count += 1
        if count > 0:
            self.rr_valid10 += valid / count

    def results(self):
        m=max(self.total,1)
        return{
            "Total Queries": self.total,
            "Baseline MRR@50": self.mrr/m,
            "Reranked MRR@50": self.rr_mrr/m,
            "Baseline Hits@1": self.hits1/m,
            "Reranked Hits@1": self.rr_hits1/m,
            "Baseline Hits@10": self.hits10/m,
            "Reranked Hits@10": self.rr_hits10/m,
            "Baseline Valid@1": self.valid1/m,
            "Reranked Valid@1": self.rr_valid1/m,
            "Baseline Candidate Valid Rate@10": self.valid10/m,
            "Reranked Candidate Valid Rate@10": self.rr_valid10/m,
            "Coverage": self.predict/m,
            "Abstention": self.abstain/m,
        }

def load_json(file,dataset):
    file.parent.mkdir(parents=True,exist_ok=True)
    with open(file,"w",encoding="utf-8") as f:
        json.dump(dataset,f,indent=2,ensure_ascii=False)

def main()->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file",type=Path,default=Path("runs/annotated_candidates/drugbank_rotate_test_tail_top50.jsonl.gz"))
    parser.add_argument("--baseline",type=Path,default=Path("runs/evaluation/drugbank_rotate_test_tail_baseline.json"))
    parser.add_argument("--controlled",type=Path,default=Path("runs/evaluation/drugbank_rotate_test_tail_simple.json"))
    args = parser.parse_args()
    evaluator=Evaluator()
    for i in iter_candidate_records(args.input_file):
        evaluator.process(i)
    results=evaluator.results()
    baseline={
        "dataset": "drugbank_core",
        "model": "rotate",
        "split": "test_tail",
        "total queries": int(results["Total Queries"]),
        "mrr at 50": results["Baseline MRR@50"],
        "hits at 1": results["Baseline Hits@1"],
        "hits at 10": results["Baseline Hits@10"],
        "valid at 1": results["Baseline Valid@1"],
        "candidate valid rate at 10": results["Baseline Candidate Valid Rate@10"],
    }
    controlled={
        "dataset": "drugbank_core",
        "model": "rotate",
        "split": "test_tail",
        "total queries": int(results["Total Queries"]),
        "mrr at 50": results["Reranked MRR@50"],
        "hits at 1": results["Reranked Hits@1"],
        "hits at 10": results["Reranked Hits@10"],
        "valid at 1": results["Reranked Valid@1"],
        "candidate valid rate at 10": results["Reranked Candidate Valid Rate@10"],
        "coverage": results["Coverage"],
        "abstention": results["Abstention"],
    }
    load_json(args.baseline,baseline)
    load_json(args.controlled,controlled)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
