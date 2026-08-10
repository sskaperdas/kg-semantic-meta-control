from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker

resources = SemanticResources("data/processed/drugbank_core")
print(resources.direct_types("drug:DB00001"))
print(resources.constraint("has_target"))

checker = SemanticChecker(resources)

valid_result = checker.check_triple(
    head="drug:DB00001",
    relation="has_target",
    tail="target:BE0000048",
)

invalid_result = checker.check_triple(
    head="drug:DB00001",
    relation="has_target",
    tail="category:proteins",
)

unknown_result = checker.check_triple(
    head="drug:DB00001",
    relation="has_target1",
    tail="target:BE0000048",
)

unknown_result1 = checker.check_triple(
    head="drug:DB00000",
    relation="has_target",
    tail="target:BE0000048",
)

print(valid_result)
print(invalid_result)
print(unknown_result)
print(unknown_result1)

assert valid_result.status == "VALID"
assert invalid_result.status == "INVALID"
assert unknown_result.status == "UNKNOWN"
assert unknown_result1.status == "UNKNOWN"

print("All semantic checker tests passed.")