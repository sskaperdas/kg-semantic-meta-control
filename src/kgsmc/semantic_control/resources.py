from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


@dataclass(frozen=True)
class RelationConstraint:
    domain: frozenset[str]
    range: frozenset[str]


class SemanticResources:
    def __init__(self, dataset_dir: str | Path):
        dataset_dir = Path(dataset_dir)

        self.entity_types = self._load_entity_types(
            read_json(dataset_dir / "entity_types.json")
        )
        self.relation_constraints = self._load_constraints(
            read_json(dataset_dir / "relation_constraints.json")
        )
        self.parents = self._load_hierarchy(
            read_json(dataset_dir / "class_hierarchy.json")
        )
        self.disjoint_pairs = self._load_disjoint_pairs(
            read_json(dataset_dir / "disjoint_pairs.json")
        )

    @staticmethod
    def _load_entity_types(raw: dict) -> dict[str, frozenset[str]]:
        return {
            str(entity): frozenset(
                [types] if isinstance(types, str) else map(str, types)
            )
            for entity, types in raw.items()
        }

    @staticmethod
    def _load_constraints(
        raw: dict,
    ) -> dict[str, RelationConstraint]:
        result = {}

        for relation, value in raw.items():
            domain = value.get("domain", [])
            range_values = value.get("range", [])

            if isinstance(domain, str):
                domain = [domain]

            if isinstance(range_values, str):
                range_values = [range_values]

            result[str(relation)] = RelationConstraint(
                domain=frozenset(map(str, domain)),
                range=frozenset(map(str, range_values)),
            )

        return result

    @staticmethod
    def _load_hierarchy(raw: dict) -> dict[str, frozenset[str]]:
        result = {}

        for child, value in raw.items():
            if isinstance(value, str):
                parents = [value]
            elif isinstance(value, list):
                parents = value
            elif isinstance(value, dict):
                parents = (
                    value.get("parents")
                    or value.get("superclasses")
                    or value.get("ancestors")
                    or []
                )
            else:
                parents = []

            result[str(child)] = frozenset(map(str, parents))

        return result

    @staticmethod
    def _load_disjoint_pairs(raw) -> frozenset[frozenset[str]]:
        pairs = set()

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list) and len(item) == 2:
                    pairs.add(frozenset(map(str, item)))

        elif isinstance(raw, dict):
            for left, right_values in raw.items():
                if isinstance(right_values, str):
                    right_values = [right_values]

                for right in right_values:
                    pairs.add(frozenset({str(left), str(right)}))

        return frozenset(pairs)

    def direct_types(self, entity: str) -> frozenset[str]:
        return self.entity_types.get(entity, frozenset())

    def constraint(
        self,
        relation: str,
    ) -> RelationConstraint | None:
        return self.relation_constraints.get(relation)

    def ancestors(self, class_name: str) -> frozenset[str]:
        visited = set()
        stack = list(self.parents.get(class_name, frozenset()))

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            stack.extend(self.parents.get(current, frozenset()))

        return frozenset(visited)

    def expanded_types(self, entity: str) -> frozenset[str]:
        direct = self.direct_types(entity)
        expanded = set(direct)

        for class_name in direct:
            expanded.update(self.ancestors(class_name))

        return frozenset(expanded)

    def are_disjoint(self, left: str, right: str) -> bool:
        return frozenset({left, right}) in self.disjoint_pairs