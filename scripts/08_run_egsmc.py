from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from kgsmc.semantic_control.candidate_io import iter_candidate_records
from kgsmc.semantic_control.gate import EvidenceGate


def bootstrap(baseline:np.ndarray,controlled:np.ndarray,num_samples:int=10000,seed:int=42):
    random_g=np.random.default_rng(seed)
    n=len(baseline)
    diff=float(np.mean(controlled)-np.mean(baseline))
    query_d=controlled-baseline
    resampled=random_g.integers(0,n,size=(num_samples,n))
    bootstrap_diff=np.mean(query_d[resampled],axis=1)
    if diff>0:
        p_value=float(np.mean(bootstrap_diff<=0))
    else:
        p_value=float(np.mean(bootstrap_diff>=0))
    p_value=min(1.0, 2.0*p_value)
    conf_interval_l=float(np.percentile(bootstrap_diff,2.5))
    conf_interval_u=float(np.percentile(bootstrap_diff,97.5))
    return{
        "mean difference": diff,
        "confidence interval 95": [conf_interval_l,conf_interval_u],
        "p value": p_value
    }


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
        self.preserve=0
        self.rerank=0
        self.abstain=0
        self.h1_list=[]
        self.rr_h1_list=[]
        self.mrr_list=[]
        self.rr_mrr_list=[]
        self.covered_h1_list = []
        self.covered_rr_h1_list = []
        self.covered_mrr_list = []
        self.covered_rr_mrr_list = []
        self.covered_mrr = 0
        self.covered_hits1 = 0
        self.covered_hits10 = 0
        self.covered_rr_mrr = 0
        self.covered_rr_hits1 = 0
        self.covered_rr_hits10 = 0
        self.gate=EvidenceGate()

    def process(self,record:dict):
        self.total+=1
        candidates=record.get("candidates",[])
        h1,mrr,h10=self.baseline_evaluate(candidates)
        decision=self.gate.decide_query(candidates)
        if decision=="PRESERVE":
            self.preserve+=1
            final=candidates
            rr_h1,rr_mrr,rr_h10=self.controlled_evaluate(final)
            self.covered(h1,mrr,h10,rr_h1,rr_mrr,rr_h10)
        elif decision=="RERANK":
            self.rerank+=1
            strong,basic,insufficient,conflict=[],[],[],[]
            for i in candidates:
                classified=self.gate.classify_candidate(i)
                if classified=="STRONG_SUPPORT":
                    strong.append(i)
                elif classified=="BASIC_SUPPORT":
                    basic.append(i)
                elif classified=="INSUFFICIENT":
                    insufficient.append(i)
                else:
                    conflict.append(i)
            final=strong+basic+insufficient+conflict
            rr_h1,rr_mrr,rr_h10 = self.controlled_evaluate(final)
            self.covered(h1,mrr,h10,rr_h1,rr_mrr,rr_h10)
        elif decision=="ABSTAIN":
            self.abstain+=1
            self.rr_h1_list.append(0)
            self.rr_mrr_list.append(0.0)

    def covered(self,h1,mrr,h10,rr_h1,rr_mrr,rr_h10):
        self.covered_h1_list.append(h1)
        self.covered_mrr_list.append(mrr)
        self.covered_mrr+=mrr
        self.covered_hits1+=h1
        self.covered_hits10+=h10
        self.covered_rr_h1_list.append(rr_h1)
        self.covered_rr_mrr_list.append(rr_mrr)
        self.covered_rr_mrr+=rr_mrr
        self.covered_rr_hits1+=rr_h1
        self.covered_rr_hits10+=rr_h10


    def baseline_evaluate(self,candidates:list):
        h1=0
        h10=0
        mrr=0.0
        for i in candidates:
            if i.get("is_gold"):
                rank = i.get("rank",999)
                mrr = 1.0 / rank
                self.mrr+=mrr
                if rank == 1:
                    h1=1
                    self.hits1 += 1
                if rank <= 10:
                    h10=1
                    self.hits10 += 1
                break
        self.h1_list.append(h1)
        self.mrr_list.append(mrr)
        if candidates:
            first=candidates[0]
            s=first.get("evidence",{}).get("semantic_status")
            if s=="VALID":
                self.valid1+=1
        valid = 0
        count = 0
        for i in candidates:
            if count == 10:
                break
            s = i.get("evidence",{}).get("semantic_status")
            if s == "VALID":
                valid += 1
            count += 1
        if count > 0:
            self.valid10 += valid / count
        return h1,mrr,h10

    def controlled_evaluate(self,candidates:list):
        new=1
        h1=0
        h10=0
        mrr=0.0
        for i in candidates:
            if i.get("is_gold"):
                mrr=1.0/new
                self.rr_mrr+=mrr
                if new==1:
                    self.rr_hits1+=1
                    h1=1
                if new<=10:
                    h10=1
                    self.rr_hits10+=1
                break
            new+=1
        self.rr_h1_list.append(h1)
        self.rr_mrr_list.append(mrr)
        if candidates:
            first=candidates[0]
            s=first.get("evidence",{}).get("semantic_status")
            if s=="VALID":
                self.rr_valid1+=1
        valid = 0
        count = 0
        for i in candidates:
            if count == 10:
                break
            s = i.get("evidence", {}).get("semantic_status")
            if s == "VALID":
                valid += 1
            count += 1
        if count > 0:
            self.rr_valid10 += valid / count
        return h1,mrr,h10

    def results(self):
        m=max(self.total,1)
        covered_m=max(self.preserve+self.rerank,1)
        h1=np.array(self.h1_list,dtype=np.float64)
        mrr=np.array(self.mrr_list,dtype=np.float64)
        rr_h1 = np.array(self.rr_h1_list, dtype=np.float64)
        rr_mrr = np.array(self.rr_mrr_list, dtype=np.float64)
        s_h1=bootstrap(h1,rr_h1)
        s_mrr=bootstrap(mrr,rr_mrr)

        covered_h1 = np.array(self.covered_h1_list, dtype=np.float64)
        covered_mrr = np.array(self.covered_mrr_list, dtype=np.float64)
        covered_rr_h1 = np.array(self.covered_rr_h1_list, dtype=np.float64)
        covered_rr_mrr = np.array(self.covered_rr_mrr_list, dtype=np.float64)
        if len(covered_h1)>0:
            covered_s_h1 = bootstrap(covered_h1, covered_rr_h1)
            covered_s_mrr = bootstrap(covered_mrr, covered_rr_mrr)
        else:
            covered_s_h1 = {}
            covered_s_mrr={}

        return{
            "Total Queries": self.total,
            "Covered Queries": self.preserve+self.rerank,
            "Baseline MRR@50": self.mrr/m,
            "EGSMC MRR@50": self.rr_mrr/m,
            "Baseline Hits@1": self.hits1/m,
            "EGSMC Hits@1": self.rr_hits1/m,
            "Baseline Hits@10": self.hits10/m,
            "EGSMC Hits@10": self.rr_hits10/m,
            "Baseline Valid@1": self.valid1/m,
            "EGSMC Valid@1": self.rr_valid1/m,
            "Baseline Candidate Valid Rate@10": self.valid10 / m,
            "EGSMC Candidate Valid Rate@10": self.rr_valid10 / m,
            "Conditional Coverage": (self.preserve+self.rerank)/m,
            "Conditional Baseline MRR@50":self.covered_mrr/covered_m,
            "Conditional EGSMC MRR@50": self.covered_rr_mrr / covered_m,
            "Conditional Baseline Hits@1": self.covered_hits1 / covered_m,
            "Conditional EGSMC Hits@1": self.covered_rr_hits1 / covered_m,
            "Conditional Baseline Hits@10": self.covered_hits10 / covered_m,
            "Conditional EGSMC Hits@10": self.covered_rr_hits10 / covered_m,
            "Decisions":{
                "Preserve": self.preserve/m,
                "Rerank": self.rerank/m,
                "Abstain": self.abstain/m,
            },
            "Coverage": (self.preserve+self.rerank)/m,
            "Abstention": self.abstain/m,
            "Significance Hits@1":s_h1,
            "Significance MRR@50":s_mrr,
            "Significance Conditional Hits@1":covered_s_h1,
            "Significance Conditional MRR@50":covered_s_mrr,
        }

def load_json(file,dataset):
    file.parent.mkdir(parents=True,exist_ok=True)
    with open(file,"w",encoding="utf-8") as f:
        json.dump(dataset,f,indent=2,ensure_ascii=False)

def main()->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file",type=Path,default=Path("runs/evidence/eurostat_rotate_test_tail_top50_evidence_v02.jsonl.gz"))
    parser.add_argument("--baseline",type=Path,default=Path("runs/evaluation/eurostat_rotate_test_tail_baseline.json"))
    parser.add_argument("--controlled",type=Path,default=Path("runs/evaluation/eurostat_rotate_test_tail_egsmc.json"))
    args = parser.parse_args()
    evaluator=Evaluator()
    for i in iter_candidate_records(args.input_file):
        evaluator.process(i)
    results=evaluator.results()
    baseline={
        "dataset": "eurostat_structural",
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
        "dataset": "eurostat structural",
        "model": "rotate",
        "split": "test_tail",
        "total queries": int(results["Total Queries"]),
        "mrr at 50": results["EGSMC MRR@50"],
        "hits at 1": results["EGSMC Hits@1"],
        "hits at 10": results["EGSMC Hits@10"],
        "valid at 1": results["EGSMC Valid@1"],
        "candidate valid rate at 10": results["EGSMC Candidate Valid Rate@10"],
        "conditional on coverage":{
            "mrr at 50": results["Conditional Baseline MRR@50"],
            "hits at 1": results["Conditional Baseline Hits@1"],
            "hits at 10": results["Conditional Baseline Hits@10"],
            "egsmc mrr at 50": results["Conditional EGSMC MRR@50"],
            "egsmc hits at 1": results["Conditional EGSMC Hits@1"],
            "egsmc hits at 10": results["Conditional EGSMC Hits@10"],
        },
        "coverage": results["Coverage"],
        "abstention": results["Abstention"],
        "significance":{
            "hits@1":results["Significance Hits@1"],
            "mrr@50":results["Significance MRR@50"],
            "conditional hits@1": results["Significance Conditional Hits@1"],
            "conditional mrr@50": results["Significance Conditional MRR@50"],
        }
    }
    load_json(args.baseline,baseline)
    load_json(args.controlled,controlled)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
