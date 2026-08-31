from __future__ import annotations

from collections import defaultdict,Counter
from pathlib import Path

from kgsmc.semantic_control.checker import SemanticChecker
from kgsmc.semantic_control.resources import SemanticResources


class GraphEvidenceIndex:
    # Απλό structural index του TRAIN graph.
    # adjacency[u][v] = set of relation labels

    def __init__(self, train_file: str | Path):
        self.train_file = Path(train_file)

        self.adjacency: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.head_role_count: Counter[tuple[str,str]] = Counter()
        self.tail_role_count: Counter[tuple[str,str]] = Counter()

        self._load()

    def _load(self) -> None:
        if not self.train_file.exists():
            raise FileNotFoundError(self.train_file)

        with self.train_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.rstrip("\r\n")

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid triple at "
                        f"{self.train_file}:{line_number}"
                    )

                head, relation, tail = parts

                # Original direction
                self.adjacency[head][tail].add(
                    relation
                )

                # Reverse traversal.
                # ^relation σημαίνει traversal ανάποδα.
                self.adjacency[tail][head].add(
                    "^" + relation
                )
                self.head_role_count[(relation,head)] += 1
                self.tail_role_count[(relation,tail)] += 1

    def _relation_for_edge(
        self,
        source: str,
        target: str,
    ) -> str | None:
        relations = self.adjacency.get(
            source,
            {},
        ).get(
            target,
            set(),
        )

        if not relations:
            return None

        return sorted(relations)[0]

    def relation_role_evidence(
            self,
            candidate:str,
            relation:str,
            query_type:str,
    )->dict:
        if query_type=="head_prediction":
            count=self.head_role_count.get((relation,candidate),0)
        elif query_type=="tail_prediction":
            count=self.tail_role_count.get((relation,candidate),0)
        else:
            raise ValueError(f"Unknown type {query_type}")
        return {
            "relation_role_supported":count>0,
            "relation_role_count": count,
        }


    def find_short_witness(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> dict:
        # Βρίσκει structural witness μήκους 1 ή 2.
        # Δεν χρησιμοποιεί το ίδιο candidate relation
        # ως direct evidence για το ίδιο triple.

        head_neighbors = self.adjacency.get(
            head,
            {}
        )

        # 1-hop witness
        if tail in head_neighbors:
            available_relations = [
                edge_relation
                for edge_relation in sorted(
                    head_neighbors[tail]
                )
                if edge_relation not in {
                    relation,
                    "^" + relation,
                }
            ]

            if available_relations:
                witness_relation = (
                    available_relations[0]
                )

                return {
                    "has_structural_support": True,
                    "shortest_witness_length": 1,
                    "witness_path": [
                        head,
                        witness_relation,
                        tail,
                    ],
                }

        # 2-hop witness
        tail_neighbors = self.adjacency.get(
            tail,
            {}
        )

        common_neighbors = (
            set(head_neighbors)
            & set(tail_neighbors)
        )

        valid=sorted(
            [node for node in common_neighbors if node!=head and node!=tail]
        )

        if valid:
            # Deterministic choice
            middle = valid[0]

            first_relation = self._relation_for_edge(
                head,
                middle,
            )

            second_relation = self._relation_for_edge(
                middle,
                tail,
            )

            if (
                first_relation is not None
                and second_relation is not None
            ):
                return {
                    "has_structural_support": True,
                    "shortest_witness_length": 2,
                    "witness_path": [
                        head,
                        first_relation,
                        middle,
                        second_relation,
                        tail,
                    ],
                }

        return {
            "has_structural_support": False,
            "shortest_witness_length": None,
            "witness_path": None,
        }

    def three_hop_witness(
            self,
            start:str,
    )->tuple[Counter[str],dict[str,list[str]]]:
        neighbors = set(self.adjacency.get(start,{}).keys())
        u=neighbors-{start}
        count: Counter[str]=Counter()
        paths:dict[str,list[str]]={}
        for i in u:
            u_neighbors=set(self.adjacency.get(i,{}).keys())
            v=u_neighbors-{start,i}
            for j in v:
                v_neighbors = set(self.adjacency.get(j, {}).keys())
                t = v_neighbors - {start, i,j}
                for k in t:
                    count[k]+=1
                    if k not in paths:
                        r1=self._relation_for_edge(start,i)
                        r2=self._relation_for_edge(i,j)
                        r3=self._relation_for_edge(j,k)
                        if r1 and r2 and r3:
                            paths[k]=[start,r1,i,r2,j,r3,k]

        return count,paths


class EvidenceExtractor:
    def __init__(
        self,
        resources: SemanticResources,
        checker: SemanticChecker,
        graph_index: GraphEvidenceIndex,
    ):
        self.resources = resources
        self.checker = checker
        self.graph_index = graph_index

    def _type_evidence(
        self,
        entity: str,
        expected_types: frozenset[str],
    ) -> dict:

        direct_types = self.resources.direct_types(
            entity
        )

        expanded_types = self.resources.expanded_types(
            entity
        )

        direct_match = bool(
            direct_types & expected_types
        )

        expanded_match = bool(
            expanded_types & expected_types
        )

        hierarchy_match = (
            expanded_match
            and not direct_match
        )

        conflicts: list[str] = []

        for actual_type in sorted(direct_types):
            for expected_type in sorted(
                expected_types
            ):
                if self.resources.are_disjoint(
                    actual_type,
                    expected_type,
                ):
                    conflicts.append(
                        f"{actual_type} disjoint "
                        f"{expected_type}"
                    )

        if not expected_types:
            status = "NO_CONSTRAINT"

        elif not direct_types:
            status = "MISSING_TYPE"

        elif direct_match and conflicts:
            status = "MIXED"

        elif hierarchy_match and conflicts:
            status = "MIXED"

        elif direct_match:
            status = "DIRECT_SUPPORT"

        elif hierarchy_match:
            status = "HIERARCHY_SUPPORT"

        elif conflicts:
            status = "DISJOINT_CONFLICT"

        else:
            status = "MISMATCH"

        return {
            "status": status,
            "direct_types": sorted(direct_types),
            "expanded_types": sorted(
                expanded_types
            ),
            "expected": sorted(expected_types),
            "direct_match": direct_match,
            "hierarchy_match": hierarchy_match,
            "disjoint_conflicts": conflicts,
        }

    def extract(
        self,
        head: str,
        relation: str,
        tail: str,
        query_type:str,
        three_hop_count:Counter[str],
        three_hop_examples:dict[str,list[str]],
    ) -> dict:

        semantic_result = (
            self.checker.check_triple(
                head,
                relation,
                tail,
            )
        )

        constraint = self.resources.constraint(
            relation
        )

        if constraint is None:
            expected_domain = frozenset()
            expected_range = frozenset()

        else:
            expected_domain = constraint.domain
            expected_range = constraint.range

        domain_evidence = self._type_evidence(
            head,
            expected_domain,
        )

        range_evidence = self._type_evidence(
            tail,
            expected_range,
        )

        short_graph_evidence = (
            self.graph_index.find_short_witness(
                head,
                relation,
                tail,
            )
        )
        if query_type=="tail_prediction":
            candidate=tail
        else:
            candidate=head

        count=0
        if three_hop_count is not None:
            count=three_hop_count.get(candidate,0)

        example=None
        if three_hop_examples is not None:
            example=three_hop_examples.get(candidate)

        three_hop_evidence={
            "has_three_hop_witness":count>0,
            "three_hop_count":count,
            "example_three_hop_path":example
        }

        role_evidence=(
            self.graph_index.relation_role_evidence(
                candidate=candidate,
                relation=relation,
                query_type=query_type,
            )
        )

        graph_evidence={
            **short_graph_evidence,
            **three_hop_evidence,
            **role_evidence,
        }

        return {
            "semantic_status": (
                semantic_result.status
            ),
            "ontology": {
                "domain": domain_evidence,
                "range": range_evidence,
            },
            "graph": graph_evidence,
        }


def add_evidence_to_record(
    record: dict,
    extractor: EvidenceExtractor,
) -> dict:

    query = record["query"]
    query_type = record["query_type"]

    if query_type=="tail_prediction":
        start=query["head"]
    elif query_type=="head_prediction":
        start=query["tail"]
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    three_hop_count,three_hop_examples=(
        extractor.graph_index.three_hop_witness(start)
    )

    for candidate in record["candidates"]:

        if query_type == "tail_prediction":
            head = query["head"]
            relation = query["relation"]
            tail = candidate["entity"]

        elif query_type == "head_prediction":
            head = candidate["entity"]
            relation = query["relation"]
            tail = query["tail"]

        else:
            raise ValueError(
                f"Unsupported query type: "
                f"{query_type}"
            )

        candidate["evidence"] = (
            extractor.extract(
                head=head,
                relation=relation,
                tail=tail,
                query_type=query_type,
                three_hop_count= three_hop_count,
                three_hop_examples=three_hop_examples,
            )
        )

    return record