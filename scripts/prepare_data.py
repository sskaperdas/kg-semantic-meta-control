"""CLI script for dataset preparation."""
#!/usr/bin/env python3
"""
Shared dataset preparation script for KG projects.

Purpose
-------
This script converts raw KG data into a clean triple dataset suitable for:
1. optional embedding model training
2. the main semantic meta-control pipeline

Supported input formats
-----------------------
- RDF graphs: .ttl, .nt, .n3, .xml, .rdf
- Tabular triples: .tsv, .csv

Outputs
-------
- train.tsv
- valid.tsv
- test.tsv
- all_triples.tsv
- entities.tsv
- relations.tsv
- stats.json

Default triple format
---------------------
head<TAB>relation<TAB>tail

Notes
-----
- By default, triples with literal objects are removed.
- rdf:type triples are kept unless excluded by predicate filters.
- Split is random but reproducible via --random-seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import Graph, Literal, URIRef

Triple = Tuple[str, str, str]

RDF_SUFFIXES = {".ttl", ".nt", ".n3", ".xml", ".rdf"}
TABULAR_SUFFIXES = {".tsv", ".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KG triples for training and semantic processing.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (.ttl/.nt/.rdf/.xml/.tsv/.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed outputs will be saved.",
    )

    parser.add_argument(
        "--format",
        default=None,
        help=(
            "Explicit input format. "
            "For RDF: turtle, nt, n3, xml. "
            "For tabular: tsv, csv. "
            "If omitted, inferred from file extension."
        ),
    )

    parser.add_argument(
        "--delimiter",
        default=None,
        help="Delimiter for tabular input. If omitted, inferred from extension.",
    )

    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set this if the input TSV/CSV has a header row.",
    )

    parser.add_argument(
        "--head-col",
        default="head",
        help="Head column name for CSV/TSV with header, or integer index (e.g. 0) for no-header tabular input.",
    )
    parser.add_argument(
        "--rel-col",
        default="relation",
        help="Relation column name for CSV/TSV with header, or integer index (e.g. 1) for no-header tabular input.",
    )
    parser.add_argument(
        "--tail-col",
        default="tail",
        help="Tail column name for CSV/TSV with header, or integer index (e.g. 2) for no-header tabular input.",
    )

    parser.add_argument(
        "--keep-literals",
        action="store_true",
        help="Keep triples whose object is a literal. By default they are removed.",
    )

    parser.add_argument(
        "--lowercase-relations",
        action="store_true",
        help="Lowercase relation strings after loading.",
    )

    parser.add_argument(
        "--strip-whitespace",
        action="store_true",
        help="Strip leading/trailing whitespace from textual fields.",
    )

    parser.add_argument(
        "--predicate-whitelist",
        default=None,
        help="Optional path to a text file with one predicate URI/name per line to keep.",
    )
    parser.add_argument(
        "--predicate-blacklist",
        default=None,
        help="Optional path to a text file with one predicate URI/name per line to exclude.",
    )

    parser.add_argument(
        "--min-relation-frequency",
        type=int,
        default=1,
        help="Minimum number of occurrences required for a relation to be kept.",
    )
    parser.add_argument(
        "--min-entity-frequency",
        type=int,
        default=1,
        help="Minimum total degree frequency required for an entity to be kept.",
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate triples after loading.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )

    parser.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort outputs lexicographically before writing.",
    )

    return parser.parse_args()


def infer_format(input_path: Path, explicit_format: Optional[str]) -> str:
    if explicit_format:
        return explicit_format.lower()

    suffix = input_path.suffix.lower()
    if suffix in RDF_SUFFIXES:
        if suffix == ".ttl":
            return "turtle"
        if suffix == ".nt":
            return "nt"
        if suffix == ".n3":
            return "n3"
        if suffix in {".xml", ".rdf"}:
            return "xml"
    if suffix in TABULAR_SUFFIXES:
        return suffix.lstrip(".")
    raise ValueError(f"Could not infer format from suffix '{suffix}'. Please pass --format explicitly.")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_list_file(path_str: Optional[str]) -> Optional[Set[str]]:
    if not path_str:
        return None
    path = Path(path_str)
    values = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                values.add(value)
    return values


def normalize_field(value: str, strip_whitespace: bool) -> str:
    return value.strip() if strip_whitespace else value


def maybe_lower_relation(value: str, lowercase_relations: bool) -> str:
    return value.lower() if lowercase_relations else value


def read_rdf_triples(
    input_path: Path,
    rdf_format: str,
    keep_literals: bool,
    strip_whitespace: bool,
    lowercase_relations: bool,
) -> List[Triple]:
    graph = Graph()
    graph.parse(str(input_path), format=rdf_format)

    triples: List[Triple] = []

    for s, p, o in graph:
        if not isinstance(s, URIRef):
            continue
        if not isinstance(p, URIRef):
            continue

        head = normalize_field(str(s), strip_whitespace)
        rel = maybe_lower_relation(normalize_field(str(p), strip_whitespace), lowercase_relations)

        if isinstance(o, URIRef):
            tail = normalize_field(str(o), strip_whitespace)
            triples.append((head, rel, tail))
        elif isinstance(o, Literal):
            if keep_literals:
                tail = normalize_field(str(o), strip_whitespace)
                triples.append((head, rel, tail))
        else:
            # Ignore blank nodes and other unsupported node types for now.
            continue

    return triples


def _resolve_column_spec(spec: str, row: Sequence[str], header_map: Optional[dict]) -> str:
    if header_map is not None:
        if spec not in header_map:
            raise KeyError(f"Column '{spec}' not found in header. Available columns: {list(header_map.keys())}")
        return row[header_map[spec]]

    try:
        idx = int(spec)
    except ValueError as exc:
        raise ValueError(
            f"Column spec '{spec}' must be an integer index when --has-header is not set."
        ) from exc

    if idx < 0 or idx >= len(row):
        raise IndexError(f"Column index {idx} out of bounds for row with {len(row)} columns.")
    return row[idx]


def read_tabular_triples(
    input_path: Path,
    delimiter: str,
    has_header: bool,
    head_col: str,
    rel_col: str,
    tail_col: str,
    strip_whitespace: bool,
    lowercase_relations: bool,
) -> List[Triple]:
    triples: List[Triple] = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        header_map = None
        if has_header:
            try:
                header = next(reader)
            except StopIteration:
                return []
            header = [h.strip() for h in header]
            header_map = {name: idx for idx, name in enumerate(header)}

        for row in reader:
            if not row:
                continue

            h = normalize_field(_resolve_column_spec(head_col, row, header_map), strip_whitespace)
            r = normalize_field(_resolve_column_spec(rel_col, row, header_map), strip_whitespace)
            t = normalize_field(_resolve_column_spec(tail_col, row, header_map), strip_whitespace)

            if not h or not r or not t:
                continue

            r = maybe_lower_relation(r, lowercase_relations)
            triples.append((h, r, t))

    return triples


def filter_predicates(
    triples: Iterable[Triple],
    whitelist: Optional[Set[str]],
    blacklist: Optional[Set[str]],
) -> List[Triple]:
    filtered: List[Triple] = []
    for h, r, t in triples:
        if whitelist is not None and r not in whitelist:
            continue
        if blacklist is not None and r in blacklist:
            continue
        filtered.append((h, r, t))
    return filtered


def deduplicate_triples(triples: Iterable[Triple]) -> List[Triple]:
    return list(dict.fromkeys(triples))


def filter_by_relation_frequency(triples: Iterable[Triple], min_relation_frequency: int) -> List[Triple]:
    relation_counts = Counter(r for _, r, _ in triples)
    return [(h, r, t) for h, r, t in triples if relation_counts[r] >= min_relation_frequency]


def compute_entity_degree_counts(triples: Iterable[Triple]) -> Counter:
    counts: Counter = Counter()
    for h, _, t in triples:
        counts[h] += 1
        counts[t] += 1
    return counts


def filter_by_entity_frequency(triples: Iterable[Triple], min_entity_frequency: int) -> List[Triple]:
    entity_counts = compute_entity_degree_counts(triples)
    return [
        (h, r, t)
        for h, r, t in triples
        if entity_counts[h] >= min_entity_frequency and entity_counts[t] >= min_entity_frequency
    ]


def validate_split_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {train_ratio} + {valid_ratio} + {test_ratio} = {total}"
        )


def split_triples(
    triples: List[Triple],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[List[Triple], List[Triple], List[Triple]]:
    validate_split_ratios(train_ratio, valid_ratio, test_ratio)

    shuffled = triples[:]
    random.Random(random_seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # test gets the remainder
    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    return train, valid, test


def write_tsv(path: Path, triples: Iterable[Triple], sort_output: bool) -> None:
    rows = list(triples)
    if sort_output:
        rows.sort()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def write_list_tsv(path: Path, values: Iterable[str], sort_output: bool) -> None:
    rows = list(values)
    if sort_output:
        rows.sort()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for value in rows:
            writer.writerow([value])


def compute_stats(
    raw_count: int,
    after_predicate_filter: int,
    after_relation_filter: int,
    after_entity_filter: int,
    final_count: int,
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> dict:
    all_triples = train + valid + test
    entities = sorted({h for h, _, _ in all_triples} | {t for _, _, t in all_triples})
    relations = sorted({r for _, r, _ in all_triples})

    relation_counts = Counter(r for _, r, _ in all_triples)
    top_relations = relation_counts.most_common(20)

    return {
        "raw_triples_loaded": raw_count,
        "after_predicate_filter": after_predicate_filter,
        "after_relation_frequency_filter": after_relation_filter,
        "after_entity_frequency_filter": after_entity_filter,
        "final_triples": final_count,
        "num_train": len(train),
        "num_valid": len(valid),
        "num_test": len(test),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "top_relations": top_relations,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    inferred_format = infer_format(input_path, args.format)

    if inferred_format in {"tsv", "csv"}:
        delimiter = args.delimiter
        if delimiter is None:
            delimiter = "\t" if inferred_format == "tsv" else ","

        triples = read_tabular_triples(
            input_path=input_path,
            delimiter=delimiter,
            has_header=args.has_header,
            head_col=args.head_col,
            rel_col=args.rel_col,
            tail_col=args.tail_col,
            strip_whitespace=args.strip_whitespace,
            lowercase_relations=args.lowercase_relations,
        )
    else:
        triples = read_rdf_triples(
            input_path=input_path,
            rdf_format=inferred_format,
            keep_literals=args.keep_literals,
            strip_whitespace=args.strip_whitespace,
            lowercase_relations=args.lowercase_relations,
        )

    raw_count = len(triples)

    if args.deduplicate:
        triples = deduplicate_triples(triples)

    whitelist = load_list_file(args.predicate_whitelist)
    blacklist = load_list_file(args.predicate_blacklist)

    triples = filter_predicates(triples, whitelist=whitelist, blacklist=blacklist)
    after_predicate_filter = len(triples)

    triples = filter_by_relation_frequency(triples, args.min_relation_frequency)
    after_relation_filter = len(triples)

    triples = filter_by_entity_frequency(triples, args.min_entity_frequency)
    after_entity_filter = len(triples)

    if not triples:
        raise RuntimeError("No triples remain after filtering. Please relax the preprocessing constraints.")

    train, valid, test = split_triples(
        triples=triples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    all_triples = train + valid + test
    entities = {h for h, _, _ in all_triples} | {t for _, _, t in all_triples}
    relations = {r for _, r, _ in all_triples}

    write_tsv(output_dir / "train.tsv", train, args.sort_output)
    write_tsv(output_dir / "valid.tsv", valid, args.sort_output)
    write_tsv(output_dir / "test.tsv", test, args.sort_output)
    write_tsv(output_dir / "all_triples.tsv", all_triples, args.sort_output)

    write_list_tsv(output_dir / "entities.tsv", entities, args.sort_output)
    write_list_tsv(output_dir / "relations.tsv", relations, args.sort_output)

    stats = compute_stats(
        raw_count=raw_count,
        after_predicate_filter=after_predicate_filter,
        after_relation_filter=after_relation_filter,
        after_entity_filter=after_entity_filter,
        final_count=len(all_triples),
        train=train,
        valid=valid,
        test=test,
    )

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[INFO] Dataset preparation completed successfully.")
    print(f"[INFO] Input file: {input_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Raw triples loaded: {raw_count}")
    print(f"[INFO] Final triples kept: {len(all_triples)}")
    print(f"[INFO] Train/Valid/Test: {len(train)}/{len(valid)}/{len(test)}")
    print(f"[INFO] Entities: {len(entities)}")
    print(f"[INFO] Relations: {len(relations)}")


if __name__ == "__main__":
    main()
