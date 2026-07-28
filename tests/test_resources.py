import os

from torch.utils.data import dataset

from src.kgsmc.semantic_control.resources import load_resources,get_types,get_constraints,is_disjoint

def test_drugbank():
    current=os.path.dirname(os.path.abspath(__file__))
    root=os.path.dirname(current)
    path=os.path.join(root,"data","processed","drugbank_core")
    if os.path.exists(path):
        resources=load_resources(path)
        print("Successfully loaded\n")
        entity="drug:DB00001"
        types=get_types(resources,entity)
        print(f"Entity: {entity} ➜ Types: {types}")
        relation="has_target"
        constraints=get_constraints(resources,relation)
        print(f"Relation: {relation} ➜ Constraints: {constraints}")
    else:
        print("No drugbank resources found")

if __name__=="__main__":
    test_drugbank()