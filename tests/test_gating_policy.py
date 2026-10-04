from kgsmc.semantic_control.gate import EvidenceGate

def candidates(status,role_supported):
    return{
        "evidence":{
        "semantic_status": status,
        "graph": {"relation_role_supported": role_supported},
      }
    }

def test_gating_policy():
    gate=EvidenceGate()
    assert gate.classify_candidate(candidates("INVALID",True))=="CONFLICT"
    assert gate.classify_candidate(candidates("INVALID",False))=="CONFLICT"
    assert gate.classify_candidate(candidates("UNKNOWN",True))=="INSUFFICIENT"
    assert gate.classify_candidate(candidates("UNKNOWN",False))=="INSUFFICIENT"
    assert gate.classify_candidate({})=="INSUFFICIENT"
    assert gate.classify_candidate({"evidence":{}})=="INSUFFICIENT"
    assert gate.classify_candidate(candidates("VALID",True))=="STRONG_SUPPORT"
    assert gate.classify_candidate(candidates("VALID",False))=="BASIC_SUPPORT"
    assert gate.classify_candidate({"evidence":{"semantic_status":"VALID"}})=="BASIC_SUPPORT"

    preserve=[
        candidates("VALID",True),
        candidates("VALID",False)
    ]
    assert gate.decide_query(preserve)=="PRESERVE"
    rerank=[
        candidates("INVALID",False),
        candidates("VALID",True)
    ]
    assert gate.decide_query(rerank)=="RERANK"
    rerank1=[
        candidates("VALID",False),
        candidates("VALID",True)
    ]
    assert gate.decide_query(rerank1)=="RERANK"
    assert gate.decide_query([])=="ABSTAIN"
    abstain=[
        candidates("INVALID",False),
        candidates("UNKNOWN",True),
        candidates("VALID",False)
    ]
    assert gate.decide_query(abstain)=="ABSTAIN"
    abstain1=[
        candidates("INVALID",False),
        candidates("INVALID",False),
    ]
    assert gate.decide_query(abstain1)=="ABSTAIN"
