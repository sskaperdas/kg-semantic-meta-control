from __future__ import annotations
from pathlib import Path
import json
from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker

def sanity_checker():
    dataset_dir=Path("data/processed/drugbank_core")
    resources=SemanticResources(dataset_dir)
    checker=SemanticChecker(resources)
    test=dataset_dir/"test.tsv"
    output_dir=Path("runs/semantic_checks")
    output_file=output_dir/"drugbank_first100.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    checked=0
    valid=0
    invalid=0
    unknown=0
    results = []
    with open(test,"r",encoding="utf-8") as f:
        for i in f:
            line=i.strip()
            if not line:
                continue
            parts= line.split("\t")
            if len(parts)<3:
                continue
            h=parts[0]
            r=parts[1]
            t=parts[2]
            result=checker.check_triple(h,r,t)
            checked+=1
            status=getattr(result,"status",None) or(
                result.get("status") if isinstance(result,dict) else str(result)
            )
            status=str(status).upper()

            if status=="VALID":
                valid+=1
            elif status=="INVALID":
                invalid+=1
            else:
                unknown+=1
            results.append(
                {"index": checked, "head": h, "relation": r, "tail": t, "result": str(result)}
            )

            if checked==100:
                break


    report={
        "checked":checked,
        "VALID":valid,
        "INVALID":invalid,
        "UNKNOWN":unknown,
        "valid percentage":f"{(valid / checked):.2%}"
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report,f,indent=2)

if __name__=="__main__":
    sanity_checker()