from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]


# =========================================================
# Generic helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def ratio(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b


def safe_split_triples(
    triples: List[Triple],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[List[Triple], List[Triple], List[Triple], dict]:
    """
    Transductive-safe split:
    - random initial split
    - move any valid/test triple to train if it introduces unseen entity/relation
    """
    total = len(triples)
    if total == 0:
        return [], [], [], {
            "initial_total": 0,
            "moved_from_valid_to_train": 0,
            "moved_from_test_to_train": 0,
        }

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

    info = {
        "initial_total": total,
        "moved_from_valid_to_train": moved_from_valid,
        "moved_from_test_to_train": moved_from_test,
    }
    return train, safe_valid, safe_test, info


# =========================================================
# XML helpers
# =========================================================

def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_children(elem, name: str):
    return [c for c in list(elem) if strip_ns(c.tag) == name]


def find_first(elem, name: str):
    for c in list(elem):
        if strip_ns(c.tag) == name:
            return c
    return None


def text_of(elem) -> str | None:
    if elem is None or elem.text is None:
        return None
    txt = elem.text.strip()
    return txt if txt else None


def child_text(elem, name: str) -> str | None:
    return text_of(find_first(elem, name))


def normalize_text_to_id(text: str, prefix: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "unknown"
    return f"{prefix}:{text}"


def get_primary_drugbank_id(drug_elem) -> str | None:
    dbids = find_children(drug_elem, "drugbank-id")
    for dbid in dbids:
        if dbid.attrib.get("primary") == "true":
            return text_of(dbid)
    if dbids:
        return text_of(dbids[0])
    return None


def drug_entity_id(drugbank_id: str) -> str:
    return f"drug:{drugbank_id}"


def partner_entity_id(kind: str, partner_id: str | None, partner_name: str | None) -> str | None:
    if partner_id:
        return f"{kind}:{partner_id}"
    if partner_name:
        return normalize_text_to_id(partner_name, kind)
    return None


# =========================================================
# DrugBank extraction
# =========================================================

def extract_targets(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "targets")
    if parent is None:
        return triples

    for target in find_children(parent, "target"):
        partner_id = child_text(target, "id")
        name = child_text(target, "name")
        target_ent = partner_entity_id("target", partner_id, name)
        if target_ent:
            triples.append((drug_id, "has_target", target_ent))
    return triples


def extract_enzymes(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "enzymes")
    if parent is None:
        return triples

    for enzyme in find_children(parent, "enzyme"):
        partner_id = child_text(enzyme, "id")
        name = child_text(enzyme, "name")
        enzyme_ent = partner_entity_id("enzyme", partner_id, name)
        if enzyme_ent:
            triples.append((drug_id, "has_enzyme", enzyme_ent))
    return triples


def extract_transporters(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "transporters")
    if parent is None:
        return triples

    for transporter in find_children(parent, "transporter"):
        partner_id = child_text(transporter, "id")
        name = child_text(transporter, "name")
        transporter_ent = partner_entity_id("transporter", partner_id, name)
        if transporter_ent:
            triples.append((drug_id, "has_transporter", transporter_ent))
    return triples


def extract_carriers(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "carriers")
    if parent is None:
        return triples

    for carrier in find_children(parent, "carrier"):
        partner_id = child_text(carrier, "id")
        name = child_text(carrier, "name")
        carrier_ent = partner_entity_id("carrier", partner_id, name)
        if carrier_ent:
            triples.append((drug_id, "has_carrier", carrier_ent))
    return triples


def extract_groups(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "groups")
    if parent is None:
        return triples

    for group in find_children(parent, "group"):
        g = text_of(group)
        if g:
            triples.append((drug_id, "in_group", normalize_text_to_id(g, "group")))
    return triples


def extract_categories(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "categories")
    if parent is None:
        return triples

    for cat in find_children(parent, "category"):
        cat_name = child_text(cat, "category")
        if cat_name:
            triples.append((drug_id, "has_category", normalize_text_to_id(cat_name, "category")))
    return triples


def extract_pathways(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "pathways")
    if parent is None:
        return triples

    for pathway in find_children(parent, "pathway"):
        pathway_name = child_text(pathway, "name")
        if pathway_name:
            triples.append((drug_id, "has_pathway", normalize_text_to_id(pathway_name, "pathway")))
    return triples


def extract_drug_interactions(drug_elem, drug_id: str) -> List[Triple]:
    triples: List[Triple] = []
    parent = find_first(drug_elem, "drug-interactions")
    if parent is None:
        return triples

    for inter in find_children(parent, "drug-interaction"):
        other_id = child_text(inter, "drugbank-id")
        other_name = child_text(inter, "name")
        if other_id:
            other_drug = drug_entity_id(other_id)
        elif other_name:
            other_drug = normalize_text_to_id(other_name, "drug_nameonly")
        else:
            other_drug = None

        if other_drug:
            triples.append((drug_id, "interacts_with", other_drug))
    return triples


# =========================================================
# Main preparation logic
# =========================================================

def prepare_drugbank_from_xml(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    include_groups: bool = False,
    include_categories: bool = False,
    include_pathways: bool = False,
    include_interactions: bool = False,
    make_interactions_symmetric: bool = False,
    min_relation_frequency: int = 1,
) -> dict:
    ensure_dir(output_dir)

    tree = ET.parse(input_path)
    root = tree.getroot()
    drugs = [c for c in list(root) if strip_ns(c.tag) == "drug"]

    raw_triples: List[Triple] = []
    extraction_counts = Counter()
    drugs_total = 0
    skipped_drugs_no_id = 0

    for drug in drugs:
        dbid = get_primary_drugbank_id(drug)
        if not dbid:
            skipped_drugs_no_id += 1
            continue

        drugs_total += 1
        d_ent = drug_entity_id(dbid)

        targets = extract_targets(drug, d_ent)
        enzymes = extract_enzymes(drug, d_ent)
        transporters = extract_transporters(drug, d_ent)
        carriers = extract_carriers(drug, d_ent)

        raw_triples.extend(targets)
        raw_triples.extend(enzymes)
        raw_triples.extend(transporters)
        raw_triples.extend(carriers)

        extraction_counts["has_target"] += len(targets)
        extraction_counts["has_enzyme"] += len(enzymes)
        extraction_counts["has_transporter"] += len(transporters)
        extraction_counts["has_carrier"] += len(carriers)

        if include_groups:
            groups = extract_groups(drug, d_ent)
            raw_triples.extend(groups)
            extraction_counts["in_group"] += len(groups)

        if include_categories:
            categories = extract_categories(drug, d_ent)
            raw_triples.extend(categories)
            extraction_counts["has_category"] += len(categories)

        if include_pathways:
            pathways = extract_pathways(drug, d_ent)
            raw_triples.extend(pathways)
            extraction_counts["has_pathway"] += len(pathways)

        if include_interactions:
            interactions = extract_drug_interactions(drug, d_ent)
            raw_triples.extend(interactions)
            extraction_counts["interacts_with"] += len(interactions)

    # Deduplicate initial triples
    triples = sorted(set(raw_triples))

    # If interactions symmetric, add reverse edges
    if include_interactions and make_interactions_symmetric:
        extra = set()
        for h, r, t in triples:
            if r == "interacts_with":
                extra.add((t, r, h))
        triples = sorted(set(triples) | extra)

    # Optional min relation frequency filtering
    rel_counts_before = Counter(r for _, r, _ in triples)
    filtered = [
        tr for tr in triples
        if rel_counts_before[tr[1]] >= min_relation_frequency
    ]

    filtered = sorted(set(filtered))
    rel_counts_after = Counter(r for _, r, _ in filtered)

    train, valid, test, split_info = safe_split_triples(
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

    # Basic entity prefix stats
    prefix_counts = Counter()
    for ent in entity_set:
        prefix = ent.split(":", 1)[0] if ":" in ent else "unknown"
        prefix_counts[prefix] += 1

    stats = {
        "dataset": "drugbank_xml",
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "config": {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "include_groups": include_groups,
            "include_categories": include_categories,
            "include_pathways": include_pathways,
            "include_interactions": include_interactions,
            "make_interactions_symmetric": make_interactions_symmetric,
            "min_relation_frequency": min_relation_frequency,
        },
        "source_summary": {
            "drugs_total_in_xml": len(drugs),
            "drugs_used": drugs_total,
            "skipped_drugs_no_id": skipped_drugs_no_id,
        },
        "counts": {
            "raw_triples_before_dedup": len(raw_triples),
            "triples_after_dedup_before_filtering": len(triples),
            "triples_after_filtering": len(filtered),
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "entities": len(entity_set),
            "relations": len(relation_set),
        },
        "split_adjustments": split_info,
        "relation_counts_before_filtering": dict(sorted(rel_counts_before.items())),
        "relation_counts_after_filtering": dict(sorted(rel_counts_after.items())),
        "extraction_counts": dict(sorted(extraction_counts.items())),
        "entity_prefix_counts": dict(sorted(prefix_counts.items())),
        "recommendation_hints": {
            "transductive_ready": True,
            "core_biomedical_graph": True,
            "has_interactions_enabled": include_interactions,
            "category_enrichment_enabled": include_categories,
            "group_enrichment_enabled": include_groups,
            "pathway_enrichment_enabled": include_pathways,
        },
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


# =========================================================
# CLI
# =========================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DrugBank full_database.xml for KGE-style training."
    )
    parser.add_argument("--input", required=True, help="Path to DrugBank full_database.xml")
    parser.add_argument("--output-dir", required=True, help="Directory to write prepared outputs")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--include-groups", action="store_true")
    parser.add_argument("--include-categories", action="store_true")
    parser.add_argument("--include-pathways", action="store_true")
    parser.add_argument("--include-interactions", action="store_true")
    parser.add_argument("--make-interactions-symmetric", action="store_true")

    parser.add_argument("--min-relation-frequency", type=int, default=1)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    prepare_drugbank_from_xml(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_groups=args.include_groups,
        include_categories=args.include_categories,
        include_pathways=args.include_pathways,
        include_interactions=args.include_interactions,
        make_interactions_symmetric=args.make_interactions_symmetric,
        min_relation_frequency=args.min_relation_frequency,
    )


if __name__ == "__main__":
    main()
