from src.kgsmc.semantic_control.resources import get_types, get_types, is_disjoint, get_constraints


def check_triple(h,r,t,resources):
    heads=get_types(resources,h)
    tails=get_types(resources,t)
    constraints=get_constraints(resources,r)
    exp_d=constraints.get("domain",[])
    exp_r=constraints.get("range",[])
    if exp_d:
        domain_ok=any(i in exp_d for i in heads)
    else:
        domain_ok=True
    if exp_r:
        range_ok=any(j in exp_r for j in tails)
    else:
        range_ok=True
    disjoin=False
    for i in heads:
        for j in tails:
            if is_disjoint(resources,i,j):
                disjoin=True
                break
        if disjoin:
            break
    valid=domain_ok and range_ok and (not disjoin)
    if valid:
        reason=f"head type {heads} matches domain; tail type {tails} matches range"
    elif disjoin:
        reason=f"disjoint conflict between head types {heads} and tail types {tails}"
    elif not domain_ok and range_ok:
        reason=f"incompatible head type {heads} for domain and tail type {tails} for range"
    elif not domain_ok:
        reason=f"expected head type {exp_d}, got {heads}"
    else:
        reason=f"expected tail type {exp_r},got {tails}"
    return{
        "valid":valid,
        "domain_ok":domain_ok,
        "range_ok":range_ok,
        "disjoint_conflict":disjoin,
        "reason":reason
    }