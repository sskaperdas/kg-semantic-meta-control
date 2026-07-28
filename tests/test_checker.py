import os
import json
from src.kgsmc.semantic_control.resources import load_resources
from src.kgsmc.semantic_control.checker import check_triple

def test_checker():
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)
    drugbank=os.path.join(root,"data","processed","drugbank_core")
    path = os.path.join(root, "data", "processed", "drugbank_core","test.txt")
    resources=load_resources(drugbank)
    checked=0
    valid=0
    invalid=0
    with open(path,"r",encoding="utf-8") as f:
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
            result=check_triple(h,r,t,resources)
            checked+=1
            if result["valid"]:
                valid+=1
            else:
                invalid+=1
            if checked==100:
                break
    report={
        "checked":checked,
        "valid":valid,
        "invalid":invalid
    }
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    test_checker()