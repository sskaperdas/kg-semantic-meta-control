from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid triple format in {path} at line {line_no}: {line!r}"
                )
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def entities_of(triples: Iterable[Triple]) -> Set[str]:
    s: Set[str] = set()
    for h, _, t in triples:
        s.add(h)
        s.add(t)
    return s


def relations_of(triples: Iterable[Triple]) -> Set[str]:
    return {r for _, r, _ in triples}


def relation_counter(triples: Iterable[Triple]) -> Counter:
    c = Counter()
    for _, r, _ in triples:
        c[r] += 1
    return c


def median(values: List[int]) -> float:
    if not values:
        return 0.0
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


def ratio(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b


def jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def top_k_counter(counter_obj: Counter, k: int) -> List[dict]:
    return [{"item": item, "count": count} for item, count in counter_obj.most_common(k)]


def degree_stats(triples: Iterable[Triple]) -> Dict[str, float]:
    out_deg = Counter()
    in_deg = Counter()

    for h, _, t in triples:
        out_deg[h] += 1
        in_deg[t] += 1

    all_nodes = set(out_deg.keys()) | set(in_deg.keys())
    if not all_nodes:
        return {
            "nodes": 0,
            "avg_out_degree": 0.0,
            "avg_in_degree": 0.0,
            "max_out_degree": 0,
            "max_in_degree": 0,
            "median_out_degree": 0.0,
            "median_in_degree": 0.0,
        }

    out_vals = sorted(out_deg.get(n, 0) for n in all_nodes)
    in_vals = sorted(in_deg.get(n, 0) for n in all_nodes)

    return {
        "nodes": len(all_nodes),
        "avg_out_degree": sum(out_vals) / len(all_nodes),
        "avg_in_degree": sum(in_vals) / len(all_nodes),
        "max_out_degree": max(out_vals),
        "max_in_degree": max(in_vals),
        "median_out_degree": median(out_vals),
        "median_in_degree": median(in_vals),
    }


def entity_prefix(ent: str) -> str:
    if ":" in ent:
        return ent.split(":", 1)[0]
    return "unknown"


def entity_prefix_counter(triples: Iterable[Triple]) -> Counter:
    c = Counter()
    for h, _, t in triples:
        c[entity_prefix(h)] += 1
        c[entity_prefix(t)] += 1
    return c


def split_entity_prefix_stats(triples: Iterable[Triple]) -> dict:
    ents = entities_of(triples)
    c = Counter(entity_prefix(e) for e in ents)
    return dict(sorted(c.items()))


def relation_presence_by_split(
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> List[dict]:
    train_counts = relation_counter(train)
    valid_counts = relation_counter(valid)
    test_counts = relation_counter(test)

    all_relations = sorted(set(train_counts) | set(valid_counts) | set(test_counts))
    rows = []
    for rel in all_relations:
        rows.append(
            {
                "relation": rel,
                "train": train_counts.get(rel, 0),
                "valid": valid_counts.get(rel, 0),
                "test": test_counts.get(rel, 0),
                "total": train_counts.get(rel, 0) + valid_counts.get(rel, 0) + test_counts.get(rel, 0),
            }
        )
    rows.sort(key=lambda x: (-x["total"], x["relation"]))
    return rows


def duplicate_stats(
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> dict:
    train_set = set(train)
    valid_set = set(valid)
    test_set = set(test)

    return {
        "within_split_duplicates": {
            "train_duplicates_removed_if_set_based": len(train) - len(train_set),
            "valid_duplicates_removed_if_set_based": len(valid) - len(valid_set),
            "test_duplicates_removed_if_set_based": len(test) - len(test_set),
        },
        "cross_split_overlap": {
            "train_valid_overlap": len(train_set & valid_set),
            "train_test_overlap": len(train_set & test_set),
            "valid_test_overlap": len(valid_set & test_set),
        },
    }


def entity_role_stats(triples: List[Triple]) -> dict:
    as_head = Counter()
    as_tail = Counter()

    for h, _, t in triples:
        as_head[h] += 1
        as_tail[t] += 1

    both = set(as_head) & set(as_tail)
    only_head = set(as_head) - set(as_tail)
    only_tail = set(as_tail) - set(as_head)

    return {
        "entities_only_as_head": len(only_head),
        "entities_only_as_tail": len(only_tail),
        "entities_as_both": len(both),
    }


def split_summary(triples: List[Triple]) -> dict:
    ents = entities_of(triples)
    rels = relations_of(triples)
    rel_counts = relation_counter(triples)

    return {
        "triples": len(triples),
        "entities": len(ents),
        "relations": len(rels),
        "degree_stats": degree_stats(triples),
        "top_relations": top_k_counter(rel_counts, 25),
        "entity_prefix_counts_from_entities": split_entity_prefix_stats(triples),
        "entity_position_prefix_counts": dict(sorted(entity_prefix_counter(triples).items())),
    }


def transductive_check(
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> dict:
    train_entities = entities_of(train)
    train_relations = relations_of(train)

    def unseen_info(split_triples: List[Triple]) -> dict:
        unseen_entities = set()
        unseen_relations = set()
        bad_triples = []

        for h, r, t in split_triples:
            bad = False
            if h not in train_entities:
                unseen_entities.add(h)
                bad = True
            if t not in train_entities:
                unseen_entities.add(t)
                bad = True
            if r not in train_relations:
                unseen_relations.add(r)
                bad = True
            if bad and len(bad_triples) < 20:
                bad_triples.append({"h": h, "r": r, "t": t})

        return {
            "unseen_entity_count": len(unseen_entities),
            "unseen_relation_count": len(unseen_relations),
            "sample_unseen_entities": sorted(list(unseen_entities))[:20],
            "sample_unseen_relations": sorted(list(unseen_relations))[:20],
            "sample_problematic_triples": bad_triples,
            "is_transductive_safe": len(unseen_entities) == 0 and len(unseen_relations) == 0,
        }

    return {
        "valid": unseen_info(valid),
        "test": unseen_info(test),
    }


def overlap_stats(
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> dict:
    train_entities = entities_of(train)
    valid_entities = entities_of(valid)
    test_entities = entities_of(test)

    train_relations = relations_of(train)
    valid_relations = relations_of(valid)
    test_relations = relations_of(test)

    return {
        "entity_overlap": {
            "valid_in_train_ratio": ratio(len(valid_entities & train_entities), len(valid_entities)),
            "test_in_train_ratio": ratio(len(test_entities & train_entities), len(test_entities)),
            "valid_train_jaccard": jaccard(valid_entities, train_entities),
            "test_train_jaccard": jaccard(test_entities, train_entities),
        },
        "relation_overlap": {
            "valid_in_train_ratio": ratio(len(valid_relations & train_relations), len(valid_relations)),
            "test_in_train_ratio": ratio(len(test_relations & train_relations), len(test_relations)),
            "valid_train_jaccard": jaccard(valid_relations, train_relations),
            "test_train_jaccard": jaccard(test_relations, train_relations),
        },
    }


def build_recommendations(
    train: List[Triple],
    valid: List[Triple],
    test: List[Triple],
) -> dict:
    trans = transductive_check(train, valid, test)
    rel_rows = relation_presence_by_split(train, valid, test)

    very_rare_relations = [row for row in rel_rows if row["total"] < 5]
    only_train_relations = [row for row in rel_rows if row["train"] > 0 and row["valid"] == 0 and row["test"] == 0]

    rel_counts_train = relation_counter(train)
    train_total = len(train)
    dominant_relation = None
    dominant_relation_ratio = 0.0

    if rel_counts_train and train_total > 0:
        rel, cnt = rel_counts_train.most_common(1)[0]
        dominant_relation = rel
        dominant_relation_ratio = cnt / train_total

    ddi_heavy = rel_counts_train.get("interacts_with", 0) > 0 and dominant_relation == "interacts_with"

    return {
        "transductive_safe_valid": trans["valid"]["is_transductive_safe"],
        "transductive_safe_test": trans["test"]["is_transductive_safe"],
        "has_cross_split_overlap": any(
            v > 0 for v in duplicate_stats(train, valid, test)["cross_split_overlap"].values()
        ),
        "many_rare_relations_total_lt_5": len(very_rare_relations) > 0,
        "rare_relation_count_total_lt_5": len(very_rare_relations),
        "relations_only_in_train_count": len(only_train_relations),
        "dominant_relation": dominant_relation,
        "dominant_relation_ratio_train": dominant_relation_ratio,
        "ddi_dominates_training": ddi_heavy,
        "suggest_relation_min_frequency_filter": len(very_rare_relations) > 0,
        "suggest_separate_ddi_experiment": ddi_heavy,
    }


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect prepared DrugBank train/valid/test triples for KGE training readiness."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing train.txt, valid.txt, test.txt"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write inspection outputs"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_path = input_dir / "train.txt"
    valid_path = input_dir / "valid.txt"
    test_path = input_dir / "test.txt"

    for p in [train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)

    report = {
        "input_dir": str(input_dir),
        "files": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
        "split_summary": {
            "train": split_summary(train),
            "valid": split_summary(valid),
            "test": split_summary(test),
            "all": split_summary(train + valid + test),
        },
        "transductive_check": transductive_check(train, valid, test),
        "overlap_stats": overlap_stats(train, valid, test),
        "duplicate_stats": duplicate_stats(train, valid, test),
        "entity_role_stats": {
            "train": entity_role_stats(train),
            "valid": entity_role_stats(valid),
            "test": entity_role_stats(test),
            "all": entity_role_stats(train + valid + test),
        },
        "relation_presence_by_split": relation_presence_by_split(train, valid, test),
        "recommendations": build_recommendations(train, valid, test),
    }

    with (output_dir / "inspection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_lines = []
    summary_lines.append("Prepared DrugBank Inspection Summary")
    summary_lines.append("=" * 40)
    summary_lines.append("")

    for split_name in ["train", "valid", "test", "all"]:
        s = report["split_summary"][split_name]
        summary_lines.append(f"[{split_name}]")
        summary_lines.append(f"triples: {s['triples']}")
        summary_lines.append(f"entities: {s['entities']}")
        summary_lines.append(f"relations: {s['relations']}")
        summary_lines.append(f"entity prefixes: {s['entity_prefix_counts_from_entities']}")
        summary_lines.append("")

    tc = report["transductive_check"]
    summary_lines.append("[transductive check]")
    summary_lines.append(f"valid safe: {tc['valid']['is_transductive_safe']}")
    summary_lines.append(f"test safe: {tc['test']['is_transductive_safe']}")
    summary_lines.append(
        f"valid unseen entities: {tc['valid']['unseen_entity_count']}, unseen relations: {tc['valid']['unseen_relation_count']}"
    )
    summary_lines.append(
        f"test unseen entities: {tc['test']['unseen_entity_count']}, unseen relations: {tc['test']['unseen_relation_count']}"
    )
    summary_lines.append("")

    rec = report["recommendations"]
    summary_lines.append("[recommendations]")
    for k, v in rec.items():
        summary_lines.append(f"{k}: {v}")
    summary_lines.append("")

    top_rel = report["split_summary"]["all"]["top_relations"][:15]
    summary_lines.append("[top relations - all]")
    for row in top_rel:
        summary_lines.append(f"{row['item']} -> {row['count']}")

    with (output_dir / "inspection_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
