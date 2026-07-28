import json
import os

def load_resources(dataset):
    types=os.path.join(dataset,"entity_types.json")
    with open(types,"r", encoding="utf-8") as f:
        entity_types=json.load(f)

    constraints=os.path.join(dataset,"relation_constraints.json")
    with open(constraints,"r", encoding="utf-8") as f:
        relation_constraints=json.load(f)

    hierarchy=os.path.join(dataset,"class_hierarchy.json")
    if os.path.exists(hierarchy):
        with open(hierarchy, "r", encoding="utf-8") as f:
            class_hierarchy = json.load(f)
    else:
        class_hierarchy={}

    disjoint=os.path.join(dataset,"disjoint_pairs.json")
    if os.path.exists(disjoint):
        with open(disjoint, "r", encoding="utf-8") as f:
            disjoint_pairs=json.load(f)
    else:
        disjoint_pairs={}
    return{
        "entity_types":entity_types,
        "relation_constraints":relation_constraints,
        "class_hierarchy":class_hierarchy,
        "disjoint_pairs":disjoint_pairs
    }

def get_types(resources,entity):
    return resources["entity_types"].get(entity, [])

def get_constraints(resources,relation):
    return resources["relation_constraints"].get(relation, {"domain":[], "range":[]})

def is_disjoint(resources,type_a, type_b):
    disjoint_pairs = resources["disjoint_pairs"]
    if type_a in disjoint_pairs and type_b in disjoint_pairs[type_a]:
        return True
    if type_b in disjoint_pairs and type_a in disjoint_pairs[type_b]:
        return True
    return False

