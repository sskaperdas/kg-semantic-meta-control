#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Eurostat TTL raw graph and produce a detailed structural report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Eurostat_KG.ttl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where inspection outputs will be written.",
    )
    parser.add_argument(
        "--format",
        default="turtle",
        help="RDF format for rdflib parse (default: turtle).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K items to report for predicates/types/namespaces.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=200,
        help="How many sample triples to export.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def node_kind(node) -> str:
    if isinstance(node, URIRef):
        return "uri"
    if isinstance(node, Literal):
        return "literal"
    if isinstance(node, BNode):
        return "bnode"
    return "other"


def shorten_uri(uri: str) -> str:
    for sep in ["#", "/"]:
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def extract_namespace(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[0] + "#"
    if "/" in uri:
        return uri.rsplit("/", 1)[0] + "/"
    return uri


def counter_to_top_list(counter: Counter, top_k: int) -> List[Dict[str, object]]:
    return [{"item": k, "count": v} for k, v in counter.most_common(top_k)]


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_tsv(path: Path, rows: Iterable[Tuple[str, ...]], header: Tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"[INFO] Loading graph from: {input_path}")
    g = Graph()
    g.parse(str(input_path), format=args.format)
    print(f"[INFO] Graph loaded successfully. Triples: {len(g)}")

    # Global counters and sets
    predicate_counter = Counter()
    predicate_uri_object_counter = Counter()
    predicate_literal_object_counter = Counter()
    predicate_bnode_object_counter = Counter()

    subject_kind_counter = Counter()
    object_kind_counter = Counter()

    subject_set = set()
    predicate_set = set()
    uri_object_set = set()
    literal_datatype_counter = Counter()
    literal_lang_counter = Counter()
    namespace_counter = Counter()

    rdf_type_counter = Counter()
    rdf_type_subjects = defaultdict(set)

    schema_counts = Counter()
    schema_samples = defaultdict(list)

    sample_rows: List[Tuple[str, str, str, str, str]] = []

    # Useful URI strings
    RDFS_SUBCLASSOF = str(RDFS.subClassOf)
    RDFS_DOMAIN = str(RDFS.domain)
    RDFS_RANGE = str(RDFS.range)
    OWL_DISJOINTWITH = str(OWL.disjointWith)

    OWL_CLASS = str(OWL.Class)
    OWL_OBJECT_PROPERTY = str(OWL.ObjectProperty)
    OWL_DATATYPE_PROPERTY = str(OWL.DatatypeProperty)
    OWL_ANNOTATION_PROPERTY = str(OWL.AnnotationProperty)
    OWL_ONTOLOGY = str(OWL.Ontology)

    for idx, (s, p, o) in enumerate(g):
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        subject_set.add(s_str)
        predicate_set.add(p_str)
        predicate_counter[p_str] += 1

        subject_kind_counter[node_kind(s)] += 1
        object_kind_counter[node_kind(o)] += 1

        if isinstance(s, URIRef):
            namespace_counter[extract_namespace(s_str)] += 1
        if isinstance(p, URIRef):
            namespace_counter[extract_namespace(p_str)] += 1
        if isinstance(o, URIRef):
            namespace_counter[extract_namespace(o_str)] += 1
            uri_object_set.add(o_str)
            predicate_uri_object_counter[p_str] += 1
        elif isinstance(o, Literal):
            predicate_literal_object_counter[p_str] += 1
            if o.datatype:
                literal_datatype_counter[str(o.datatype)] += 1
            if o.language:
                literal_lang_counter[o.language] += 1
        elif isinstance(o, BNode):
            predicate_bnode_object_counter[p_str] += 1

        # rdf:type distribution
        if p == RDF.type:
            rdf_type_counter[o_str] += 1
            rdf_type_subjects[o_str].add(s_str)

        # schema / ontology-like counts
        if p_str == RDFS_SUBCLASSOF:
            schema_counts["rdfs:subClassOf"] += 1
            if len(schema_samples["rdfs:subClassOf"]) < 10:
                schema_samples["rdfs:subClassOf"].append((s_str, p_str, o_str))
        elif p_str == RDFS_DOMAIN:
            schema_counts["rdfs:domain"] += 1
            if len(schema_samples["rdfs:domain"]) < 10:
                schema_samples["rdfs:domain"].append((s_str, p_str, o_str))
        elif p_str == RDFS_RANGE:
            schema_counts["rdfs:range"] += 1
            if len(schema_samples["rdfs:range"]) < 10:
                schema_samples["rdfs:range"].append((s_str, p_str, o_str))
        elif p_str == OWL_DISJOINTWITH:
            schema_counts["owl:disjointWith"] += 1
            if len(schema_samples["owl:disjointWith"]) < 10:
                schema_samples["owl:disjointWith"].append((s_str, p_str, o_str))

        # ontology-related rdf:type categories
        if p == RDF.type and o_str == OWL_CLASS:
            schema_counts["rdf:type owl:Class"] += 1
        elif p == RDF.type and o_str == OWL_OBJECT_PROPERTY:
            schema_counts["rdf:type owl:ObjectProperty"] += 1
        elif p == RDF.type and o_str == OWL_DATATYPE_PROPERTY:
            schema_counts["rdf:type owl:DatatypeProperty"] += 1
        elif p == RDF.type and o_str == OWL_ANNOTATION_PROPERTY:
            schema_counts["rdf:type owl:AnnotationProperty"] += 1
        elif p == RDF.type and o_str == OWL_ONTOLOGY:
            schema_counts["rdf:type owl:Ontology"] += 1

        if len(sample_rows) < args.sample_limit:
            sample_rows.append((
                s_str,
                p_str,
                o_str,
                node_kind(s),
                node_kind(o),
            ))

    top_types = []
    for type_uri, count in rdf_type_counter.most_common(args.top_k):
        top_types.append({
            "type_uri": type_uri,
            "type_short": shorten_uri(type_uri),
            "count": count,
            "unique_subjects": len(rdf_type_subjects[type_uri]),
        })

    report = {
        "input_file": str(input_path),
        "triples_total": len(g),
        "unique_subjects": len(subject_set),
        "unique_predicates": len(predicate_set),
        "unique_uri_objects": len(uri_object_set),
        "subject_kind_counts": dict(subject_kind_counter),
        "object_kind_counts": dict(object_kind_counter),
        "top_predicates": counter_to_top_list(predicate_counter, args.top_k),
        "top_predicates_with_uri_objects": counter_to_top_list(predicate_uri_object_counter, args.top_k),
        "top_predicates_with_literal_objects": counter_to_top_list(predicate_literal_object_counter, args.top_k),
        "top_predicates_with_bnode_objects": counter_to_top_list(predicate_bnode_object_counter, args.top_k),
        "top_namespaces": counter_to_top_list(namespace_counter, args.top_k),
        "top_rdf_types": top_types,
        "top_literal_datatypes": counter_to_top_list(literal_datatype_counter, args.top_k),
        "top_literal_languages": counter_to_top_list(literal_lang_counter, args.top_k),
        "schema_counts": dict(schema_counts),
        "schema_samples": {
            k: [{"s": s, "p": p, "o": o} for s, p, o in v]
            for k, v in schema_samples.items()
        },
        "recommendation_hints": {
            "has_many_literals": object_kind_counter["literal"] > 0,
            "has_schema_triples": any(schema_counts.values()),
            "needs_predicate_review": True,
            "candidate_for_direct_prepare_data": (
                object_kind_counter["uri"] > 0 and len(predicate_set) > 0
            ),
        },
    }

    # Write JSON report
    write_json(output_dir / "eurostat_inspection_report.json", report)

    # Write samples
    write_tsv(
        output_dir / "sample_triples.tsv",
        sample_rows,
        header=("subject", "predicate", "object", "subject_kind", "object_kind"),
    )

    # Write top predicates TSV
    write_tsv(
        output_dir / "top_predicates.tsv",
        [(item["item"], str(item["count"])) for item in report["top_predicates"]],
        header=("predicate", "count"),
    )

    # Write top rdf types TSV
    write_tsv(
        output_dir / "top_rdf_types.tsv",
        [(x["type_uri"], x["type_short"], str(x["count"]), str(x["unique_subjects"])) for x in top_types],
        header=("type_uri", "type_short", "count", "unique_subjects"),
    )

    # Console summary
    print("\n[SUMMARY]")
    print(f"Triples total           : {report['triples_total']}")
    print(f"Unique subjects         : {report['unique_subjects']}")
    print(f"Unique predicates       : {report['unique_predicates']}")
    print(f"Unique URI objects      : {report['unique_uri_objects']}")
    print(f"Object kinds            : {report['object_kind_counts']}")
    print(f"Schema counts           : {report['schema_counts']}")
    print(f"Report JSON             : {output_dir / 'eurostat_inspection_report.json'}")
    print(f"Sample triples TSV      : {output_dir / 'sample_triples.tsv'}")
    print(f"Top predicates TSV      : {output_dir / 'top_predicates.tsv'}")
    print(f"Top rdf types TSV       : {output_dir / 'top_rdf_types.tsv'}")

    print("\n[INTERPRETATION HINTS]")
    print("- Αν υπάρχουν πάρα πολλά literal objects, μάλλον θα χρειαστεί filtering για training.")
    print("- Αν υπάρχουν πολλά schema triples (domain/range/subClassOf/owl:*), πιθανόν να χρειαστείς διάκριση graph vs ontology.")
    print("- Κοίτα πρώτα τα top predicates πριν αποφασίσεις αν το prepare_data.py θα τρέξει αυτούσιο ή με whitelist/blacklist.")


if __name__ == "__main__":
    main()
