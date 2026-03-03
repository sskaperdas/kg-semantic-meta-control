from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

Triple = Tuple[str, str, str]


# =========================================================
# Generic helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_uri_node(x) -> bool:
    return isinstance(x, URIRef)


def guess_rdf_format(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".ttl":
        return "turtle"
    if suffix == ".nt":
        return "nt"
    if suffix == ".n3":
        return "n3"
    if suffix == ".xml" or suffix == ".rdf":
        return "xml"
    return None


def load_graph(input_path: Path) -> Graph:
    g = Graph()
    rdf_format = guess_rdf_format(input_path)
    g.parse(str(input_path), format=rdf_format)
    return g


def graph_to_uri_triples(graph: Graph) -> List[Triple]:
    triples: List[Triple] = []
    for s, p, o in graph:
        if is_uri_node(s) and is_uri_node(p) and is_uri_node(o):
            triples.append((str(s), str(p), str(o)))
    return triples


def write_triples(triples: Iterable[Triple], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def write_vocab(items: Iterable[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x in sorted(set(items)):
            f.write(f"{x}\n")


def write_id_map(items: Iterable[str], path: Path) -> None:
    items_sorted = sorted(set(items))
    with path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items_sorted):
            f.write(f"{item}\t{idx}\n")


def safe_split_triples(
    triples: List[Triple],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[List[Triple], List[Triple], List[Triple]]:
    """
    Transductive split:
    - first do random split
    - then move any valid/test triple to train if it introduces unseen entity/relation
    """
    total = len(triples)
    if total == 0:
        return [], [], []

    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.0")

    rng = random.Random(seed)
    shuffled = list(triples)
    rng.shuffle(shuffled)

    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)
    n_test = total - n_train - n_valid

    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:n_train + n_valid + n_test]

    def entities(trs: Iterable[Triple]) -> Set[str]:
        out = set()
        for h, _, t in trs:
            out.add(h)
            out.add(t)
        return out

    def relations(trs: Iterable[Triple]) -> Set[str]:
        return {r for _, r, _ in trs}

    train_entities = entities(train)
    train_relations = relations(train)

    safe_valid: List[Triple] = []
    safe_test: List[Triple] = []

    moved_from_valid = 0
    moved_from_test = 0

    for triple in valid:
        h, r, t = triple
        if h in train_entities and t in train_entities and r in train_relations:
            safe_valid.append(triple)
        else:
            train.append(triple)
            train_entities.update([h, t])
            train_relations.add(r)
            moved_from_valid += 1

    for triple in test:
        h, r, t = triple
        if h in train_entities and t in train_entities and r in train_relations:
            safe_test.append(triple)
        else:
            train.append(triple)
            train_entities.update([h, t])
            train_relations.add(r)
            moved_from_test += 1

    return train, safe_valid, safe_test


def relation_counter(triples: Iterable[Triple]) -> Counter:
    c = Counter()
    for _, r, _ in triples:
        c[r] += 1
    return c


def type_counter(triples: Iterable[Triple]) -> Counter:
    c = Counter()
    rdf_type = str(RDF.type)
    for _, r, t in triples:
        if r == rdf_type:
            c[t] += 1
    return c


# =========================================================
# Eurostat preprocessing
# =========================================================

def preprocess_eurostat_training(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    include_rdf_type: bool = True,
    include_schema: bool = True,
) -> dict:
    """
    Eurostat training preprocessing:
    - load RDF
    - keep URI->URI triples only
    - optionally keep/remove rdf:type
    - optionally keep/remove schema relations
    - deduplicate
    - safe train/valid/test split
    - write vocab + mappings + stats

    This is specifically for KGE-style training preprocessing.
    """
    ensure_dir(output_dir)

    g = load_graph(input_path)
    all_uri_triples = graph_to_uri_triples(g)

    rdf_type = str(RDF.type)
    schema_relations = {
        str(RDFS.subClassOf),
        str(RDFS.subPropertyOf),
        str(RDFS.domain),
        str(RDFS.range),
        str(OWL.equivalentClass),
        str(OWL.equivalentProperty),
        str(OWL.inverseOf),
        str(OWL.disjointWith),
        str(OWL.sameAs),
        str(OWL.imports),
    }

    filtered: List[Triple] = []
    removal_stats = Counter()

    for h, r, t in all_uri_triples:
        if (not include_rdf_type) and (r == rdf_type):
            removal_stats["removed_rdf_type"] += 1
            continue

        if (not include_schema) and (r in schema_relations):
            removal_stats["removed_schema_relation"] += 1
            continue

        filtered.append((h, r, t))

    # deduplicate while preserving deterministic order as much as possible
    filtered = sorted(set(filtered))

    train, valid, test = safe_split_triples(
        filtered,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    entity_set = set()
    relation_set = set()
    for h, r, t in filtered:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)

    # Write outputs
    write_triples(filtered, output_dir / "all_triples.txt")
    write_triples(train, output_dir / "train.txt")
    write_triples(valid, output_dir / "valid.txt")
    write_triples(test, output_dir / "test.txt")

    write_vocab(entity_set, output_dir / "entities.txt")
    write_vocab(relation_set, output_dir / "relations.txt")
    write_id_map(entity_set, output_dir / "entity2id.txt")
    write_id_map(relation_set, output_dir / "relation2id.txt")

    stats = {
        "dataset": "eurostat",
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "config": {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "include_rdf_type": include_rdf_type,
            "include_schema": include_schema,
        },
        "counts": {
            "rdf_graph_total_triples": len(g),
            "uri_uri_triples_before_filtering": len(all_uri_triples),
            "training_triples_after_filtering": len(filtered),
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "entities": len(entity_set),
            "relations": len(relation_set),
        },
        "removals": dict(removal_stats),
        "top_relations": relation_counter(filtered).most_common(25),
        "top_rdf_types": type_counter(filtered).most_common(25),
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


# =========================================================
# DrugBank preprocessing
# =========================================================

def preprocess_drugbank_training(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Placeholder for DrugBank preprocessing.

    To be completed later
    """
    ensure_dir(output_dir)

    stats = {
        "dataset": "drugbank",
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "status": "to_be_completed",
        "notes": [
            "DrugBank preprocessing has not been implemented yet.",
            "Add source-specific parsing and filtering rules here.",
            "Keep this function separate from Eurostat to avoid mixing assumptions.",
        ],
        "config": {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


# =========================================================
# CLI
# =========================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RDF datasets for KGE-style training."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["eurostat", "drugbank"],
        help="Dataset-specific preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to RDF input file."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write processed outputs."
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--exclude-rdf-type",
        action="store_true",
        help="Exclude rdf:type triples from Eurostat training preprocessing."
    )
    parser.add_argument(
        "--exclude-schema",
        action="store_true",
        help="Exclude schema triples like subClassOf/domain/range from Eurostat training preprocessing."
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.dataset == "eurostat":
        stats = preprocess_eurostat_training(
            input_path=input_path,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            include_rdf_type=not args.exclude_rdf_type,
            include_schema=not args.exclude_schema,
        )
    elif args.dataset == "drugbank":
        stats = preprocess_drugbank_training(
            input_path=input_path,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()