from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from kgsmc.semantic_control.candidate_io import iter_candidate_records

def statistics(stat: list[int|float]) -> dict[str,float]:
    if not stat:
        return {"median":0.0,"p90":0.0}
    array=np.array(stat)
    return {
        "median":float(np.median(array)),
        "p90":float(np.percentile(array,90)),
    }

def analyze(file:Path)->dict:
    count=0
    gold=0
    nongold=0
    ghop_support=0
    nghop_support=0
    ghop_count=[]
    nghop_count=[]
    grole_support=0
    ngrole_support=0
    grole_count=[]
    ngrole_count=[]

    for i in iter_candidate_records(file):
        count+=1
        candidates=i.get("candidates",[])
        for j in candidates:
            is_gold=j.get("is_gold",False)
            evidence=j.get("evidence",{}).get("graph",{})
            three_hop=evidence.get("has_three_hop_witness",False)
            three_hop_count=evidence.get("three_hop_count",0)
            role=evidence.get("relation_role_supported",False)
            role_count=evidence.get("relation_role_count",0)
            if is_gold:
                gold+=1
                if three_hop:
                    ghop_support+=1
                ghop_count.append(three_hop_count)
                if role:
                    grole_support+=1
                grole_count.append(role_count)
            else:
                nongold+=1
                if three_hop:
                    nghop_support += 1
                nghop_count.append(three_hop_count)
                if role:
                    ngrole_support += 1
                ngrole_count.append(role_count)

    if gold > 0:
        ghop_rate = ghop_support / gold
        grole_rate = grole_support / gold
    else:
        ghop_rate = 0.0
        grole_rate = 0.0
    if nongold > 0:
        nghop_rate = nghop_support / nongold
        ngrole_rate = ngrole_support / nongold
    else:
        nghop_rate = 0.0
        ngrole_rate = 0.0

    ghop_stat = statistics(ghop_count)
    grole_stat = statistics(grole_count)
    nghop_stat = statistics(nghop_count)
    ngrole_stat = statistics(ngrole_count)

    return {
        "gold_candidates": gold,
        "nongold_candidates": nongold,
        "3_hop_support_rate": {
            "gold": ghop_rate,
            "nongold": nghop_rate,
        },
        "3_hop_witness_count": {
            "gold_median": ghop_stat["median"],
            "nongold_median": nghop_stat["median"],
            "gold_p90": ghop_stat["p90"],
            "nongold_p90": nghop_stat["p90"],
        },
        "relation_role_support": {
            "gold": grole_rate,
            "nongold": ngrole_rate,
        },
        "relation_role_count": {
            'gold_median': grole_stat['median'],
            'nongold_median': ngrole_stat['median'],
        },
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("runs/evidence/drugbank_rotate_valid_tail_top50_evidence_v02.jsonl.gz"),
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("runs/analysis/drugbank_rotate_validation_analysis.json"),
    )
    args = parser.parse_args()
    result=analyze(args.input_file)
    args.output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_file,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)


if __name__ == "__main__":
    main()
