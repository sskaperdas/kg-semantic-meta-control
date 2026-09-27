class EvidenceGate:
    def classify_candidate(self,candidate):
        evidence=candidate.get("evidence",{})
        semantic_status=evidence.get("semantic_status")
        if semantic_status=="INVALID":
            return "CONFLICT"
        elif semantic_status=="UNKNOWN":
            return "INSUFFICIENT"
        elif semantic_status=="VALID":
            has_role_support=evidence.get("graph",{}).get("relation_role_supported",False)
            if has_role_support:
                return "STRONG_SUPPORT"
            return "BASIC_SUPPORT"
        return "INSUFFICIENT"

    def decide_query(self,candidates):
        if not candidates:
            return"ABSTAIN"
        first=self.classify_candidate(candidates[0])
        if first=="STRONG_SUPPORT":
            return "PRESERVE"
        for i in candidates[1:]:
            if self.classify_candidate(i)=="STRONG_SUPPORT":
                return "RERANK"
        return "ABSTAIN"