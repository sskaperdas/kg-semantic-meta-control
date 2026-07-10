from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Set

try:
    from rdflib import Graph, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, OWL
except ImportError:
    Graph = None
    URIRef = None
    BNode = None
    RDF = None
    RDFS = None
    OWL = None


# =========================================================
# IO helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_id_map(path: Path) -> Dict[str, int]:
    """
    Supports both:
      - JSON mapping: {"entity": 0, ...}
      - TSV mapping:  entity<TAB>id
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing id map: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()}

    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid id-map line in {path} at {line_no}: {line!r}")
            key, idx = parts
            mapping[key] = int(idx)
    return mapping


def find_id_map(processed_dir: Path, stem: str) -> Path:
    for suffix in [".json", ".txt", ".tsv"]:
        p = processed_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {stem}.json/.txt/.tsv in {processed_dir}")


def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if "/" in uri:
        return uri.rstrip("/").rsplit("/", 1)[-1]
    if ":" in uri:
        return uri.split(":", 1)[0]
    return uri


# =========================================================
# DrugBank sidecars
# =========================================================

DRUGBANK_PREFIX_TO_TYPE = {
    "drug": "Drug",
    "target": "Target",
    "enzyme": "Enzyme",
    "transporter": "Transporter",
    "carrier": "Carrier",
    "category": "Category",
    "group": "Group",
    "pathway": "Pathway",
}

DRUGBANK_RELATION_CONSTRAINTS = {
    "has_target": {
        "domain": ["Drug"],
        "range": ["Target"],
    },
    "has_enzyme": {
        "domain": ["Drug"],
        "range": ["Enzyme"],
    },
    "has_transporter": {
        "domain": ["Drug"],
        "range": ["Transporter"],
    },
    "has_carrier": {
        "domain": ["Drug"],
        "range": ["Carrier"],
    },
    "has_category": {
        "domain": ["Drug"],
        "range": ["Category"],
    },
    "in_group": {
        "domain": ["Drug"],
        "range": ["Group"],
    },
    "has_pathway": {
        "domain": ["Drug"],
        "range": ["Pathway"],
    },
}


def drugbank_entity_type(entity: str) -> str:
    prefix = entity.split(":", 1)[0] if ":" in entity else "unknown"
    return DRUGBANK_PREFIX_TO_TYPE.get(prefix, "Unknown")


def build_drugbank_sidecars(processed_dir: Path, output_dir: Path) -> dict:
    ensure_dir(output_dir)

    entity2id = read_id_map(find_id_map(processed_dir, "entity2id"))
    relation2id = read_id_map(find_id_map(processed_dir, "relation2id"))

    entity_types = {}
    unknown_entities = []

    for entity in sorted(entity2id.keys()):
        typ = drugbank_entity_type(entity)
        entity_types[entity] = [typ]
        if typ == "Unknown":
            unknown_entities.append(entity)

    # Keep only constraints for relations that exist in this processed dataset.
    relation_constraints = {
        rel: DRUGBANK_RELATION_CONSTRAINTS[rel]
        for rel in sorted(relation2id.keys())
        if rel in DRUGBANK_RELATION_CONSTRAINTS
    }

    all_classes = sorted(set(DRUGBANK_PREFIX_TO_TYPE.values()))
    disjoint_pairs = [
        [a, b]
        for a, b in combinations(all_classes, 2)
    ]

    class_hierarchy = {
        cls: []
        for cls in all_classes
    }

    write_json(entity_types, output_dir / "entity_types.json")
    write_json(relation_constraints, output_dir / "relation_constraints.json")
    write_json(disjoint_pairs, output_dir / "disjoint_pairs.json")
    write_json(class_hierarchy, output_dir / "class_hierarchy.json")

    report = {
        "dataset": "drugbank",
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "num_entity_types": len(entity_types),
        "num_relation_constraints": len(relation_constraints),
        "num_disjoint_pairs": len(disjoint_pairs),
        "unknown_entities": len(unknown_entities),
        "sample_unknown_entities": unknown_entities[:20],
        "relation_constraints": relation_constraints,
        "notes": [
            "Entity types are inferred from entity prefixes.",
            "Relation constraints are deterministic for the DrugBank core extraction schema.",
        ],
    }

    write_json(report, output_dir / "semantic_sidecars_report.json")
    return report


# =========================================================
# Eurostat sidecars
# =========================================================

def require_rdflib() -> None:
    if Graph is None:
        raise RuntimeError("rdflib is required for Eurostat sidecars. Install with: pip install rdflib")


def is_xsd_datatype(uri: str) -> bool:
    return uri.startswith("http://www.w3.org/2001/XMLSchema#")


def rdf_list_items(graph: Graph, node) -> List[str]:
    """
    Resolve simple RDF lists used in owl:unionOf.
    Returns URI item strings only.
    """
    items: List[str] = []
    current = node

    while current and current != RDF.nil:
        first = graph.value(current, RDF.first)
        rest = graph.value(current, RDF.rest)

        if isinstance(first, URIRef):
            items.append(str(first))

        if rest is None or rest == RDF.nil:
            break

        current = rest

    return items


def resolve_class_expression(graph: Graph, obj) -> List[str]:
    """
    Resolve:
      - URI class
      - blank node with owl:unionOf list
    Skips datatype ranges.
    """
    out: List[str] = []

    if isinstance(obj, URIRef):
        obj_str = str(obj)
        if not is_xsd_datatype(obj_str):
            out.append(obj_str)
        return out

    if isinstance(obj, BNode):
        union_node = graph.value(obj, OWL.unionOf)
        if union_node is not None:
            out.extend(rdf_list_items(graph, union_node))

    return out


def closure_over_parents(cls: str, parents: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(parents.get(cls, set()))

    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(parents.get(x, set()) - seen)

    return seen


def build_eurostat_sidecars(
        processed_dir: Path,
        output_dir: Path,
        raw_ttl: Path,
        rdf_format: str = "turtle",
) -> dict:
    require_rdflib()
    ensure_dir(output_dir)

    entity2id = read_id_map(find_id_map(processed_dir, "entity2id"))
    relation2id = read_id_map(find_id_map(processed_dir, "relation2id"))

    entity_set = set(entity2id.keys())
    relation_set = set(relation2id.keys())

    graph = Graph()
    graph.parse(str(raw_ttl), format=rdf_format)

    # -----------------------------
    # Class hierarchy
    # -----------------------------
    parents: Dict[str, Set[str]] = {}

    for s, _, o in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            parents.setdefault(str(s), set()).add(str(o))

    class_hierarchy = {
        cls: sorted(list(ps))
        for cls, ps in sorted(parents.items())
    }

    # -----------------------------
    # Entity types
    # -----------------------------
    asserted_types: Dict[str, Set[str]] = {}

    for s, _, o in graph.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue

        s_str = str(s)
        o_str = str(o)

        if s_str not in entity_set:
            continue

        asserted_types.setdefault(s_str, set()).add(o_str)

    entity_types: Dict[str, List[str]] = {}
    entity_asserted_types: Dict[str, List[str]] = {}

    for ent in sorted(entity_set):
        direct = asserted_types.get(ent, set())
        expanded = set(direct)

        for cls in direct:
            expanded.update(closure_over_parents(cls, parents))

        entity_asserted_types[ent] = sorted(direct)
        entity_types[ent] = sorted(expanded)

    # -----------------------------
    # Relation constraints
    # -----------------------------
    relation_constraints = {}

    for rel in sorted(relation_set):
        rel_uri = URIRef(rel)

        domains: Set[str] = set()
        ranges: Set[str] = set()

        for obj in graph.objects(rel_uri, RDFS.domain):
            domains.update(resolve_class_expression(graph, obj))

        for obj in graph.objects(rel_uri, RDFS.range):
            ranges.update(resolve_class_expression(graph, obj))

        if domains or ranges:
            relation_constraints[rel] = {
                "domain": sorted(domains),
                "range": sorted(ranges),
                "domain_short": [local_name(x) for x in sorted(domains)],
                "range_short": [local_name(x) for x in sorted(ranges)],
            }

    # -----------------------------
    # Disjoint pairs
    # -----------------------------
    disjoint_pairs = []

    for s, _, o in graph.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            disjoint_pairs.append([str(s), str(o)])

    # -----------------------------
    # Write
    # -----------------------------
    write_json(entity_types, output_dir / "entity_types.json")
    write_json(entity_asserted_types, output_dir / "entity_asserted_types.json")
    write_json(relation_constraints, output_dir / "relation_constraints.json")
    write_json(disjoint_pairs, output_dir / "disjoint_pairs.json")
    write_json(class_hierarchy, output_dir / "class_hierarchy.json")

    typed_entities = sum(1 for v in entity_types.values() if v)
    constrained_relations = len(relation_constraints)

    unconstrained_relations = sorted([
        r for r in relation_set
        if r not in relation_constraints
    ])

    report = {
        "dataset": "eurostat",
        "processed_dir": str(processed_dir),
        "raw_ttl": str(raw_ttl),
        "output_dir": str(output_dir),
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "typed_entities": typed_entities,
        "typed_entity_ratio": typed_entities / max(len(entity2id), 1),
        "num_relation_constraints": constrained_relations,
        "unconstrained_relations": unconstrained_relations,
        "num_disjoint_pairs": len(disjoint_pairs),
        "num_class_hierarchy_entries": len(class_hierarchy),
        "sample_relation_constraints": dict(list(relation_constraints.items())[:10]),
        "notes": [
            "Entity types are extracted from rdf:type in the full Eurostat raw graph.",
            "Types are expanded using rdfs:subClassOf closure.",
            "Relation constraints are extracted from rdfs:domain and rdfs:range.",
            "The structural training graph intentionally excludes rdf:type and schema triples.",
        ],
    }

    write_json(report, output_dir / "semantic_sidecars_report.json")
    return report


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic sidecars for post-hoc semantic checking."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["drugbank", "eurostat"],
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Official processed training dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write sidecars. Default: processed-dir.",
    )
    parser.add_argument(
        "--raw-eurostat",
        default=None,
        help="Path to raw Eurostat TTL. Required for --dataset eurostat.",
    )
    parser.add_argument(
        "--rdf-format",
        default="turtle",
        help="RDF format for Eurostat raw graph.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir

    if args.dataset == "drugbank":
        report = build_drugbank_sidecars(
            processed_dir=processed_dir,
            output_dir=output_dir,
        )

    elif args.dataset == "eurostat":
        if not args.raw_eurostat:
            raise ValueError("--raw-eurostat is required for Eurostat sidecars")

        report = build_eurostat_sidecars(
            processed_dir=processed_dir,
            output_dir=output_dir,
            raw_ttl=Path(args.raw_eurostat),
            rdf_format=args.rdf_format,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()