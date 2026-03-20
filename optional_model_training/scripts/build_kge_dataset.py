from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Set

Triple = Tuple[str, str, str]


# =========================================================
# IO
# =========================================================

def read_triples(path: Path) -> List[Triple]:
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid triple at line {i}: {line}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def write_triples(triples: List[Triple], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def write_mapping(mapping: Dict[str, int], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for k, v in mapping.items():
            f.write(f"{k}\t{v}\n")


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# Cleaning
# =========================================================

def is_literal(x: str) -> bool:
    # simple heuristic
    return x.startswith('"') or "^^" in x


def clean_triples(
    triples: List[Triple],
    remove_literals: bool,
    remove_self_loops: bool,
) -> List[Triple]:
    cleaned = []

    for h, r, t in triples:
        if remove_literals and (is_literal(h) or is_literal(t)):
            continue
        if remove_self_loops and h == t:
            continue
        cleaned.append((h, r, t))

    return cleaned


# =========================================================
# Split (TRANSUCTIVE SAFE)
# =========================================================

def split_transductive(
    triples: List[Triple],
    train_ratio=0.8,
    valid_ratio=0.1,
    seed=42,
):
    random.seed(seed)
    triples = triples.copy()
    random.shuffle(triples)

    train = []
    valid = []
    test = []

    train_entities: Set[str] = set()

    # first pass: build train
    for h, r, t in triples:
        if len(train) < int(len(triples) * train_ratio):
            train.append((h, r, t))
            train_entities.add(h)
            train_entities.add(t)

    # second pass: assign valid/test only if entities exist
    remaining = triples[len(train):]

    for h, r, t in remaining:
        if h in train_entities and t in train_entities:
            if len(valid) < int(len(triples) * valid_ratio):
                valid.append((h, r, t))
            else:
                test.append((h, r, t))
        else:
            # fallback: keep in train
            train.append((h, r, t))
            train_entities.add(h)
            train_entities.add(t)

    return train, valid, test


# =========================================================
# Mapping
# =========================================================

def build_mappings(triples: List[Triple]):
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    return entity2id, relation2id


# =========================================================
# Stats (paper-ready)
# =========================================================

def compute_stats(triples: List[Triple]):
    entities = set()
    relations = set()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    rel_counts = Counter(r for _, r, _ in triples)

    return {
        "num_triples": len(triples),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "avg_triples_per_relation": sum(rel_counts.values()) / max(len(rel_counts), 1),
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Build KGE-ready dataset (EurostatKG / DBpedia)")

    parser.add_argument("--input", required=True, help="Input triples file (TSV)")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--remove-literals", action="store_true")
    parser.add_argument("--remove-self-loops", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading triples...")
    triples = read_triples(input_path)

    print("Cleaning...")
    triples = clean_triples(
        triples,
        remove_literals=args.remove_literals,
        remove_self_loops=args.remove_self_loops,
    )

    print("Splitting (transductive)...")
    train, valid, test = split_transductive(triples, seed=args.seed)

    print("Building mappings...")
    entity2id, relation2id = build_mappings(train + valid + test)

    print("Saving...")
    write_triples(train, output_dir / "train.txt")
    write_triples(valid, output_dir / "valid.txt")
    write_triples(test, output_dir / "test.txt")

    write_mapping(entity2id, output_dir / "entity2id.txt")
    write_mapping(relation2id, output_dir / "relation2id.txt")

    stats = {
        "train": compute_stats(train),
        "valid": compute_stats(valid),
        "test": compute_stats(test),
    }

    save_json(stats, output_dir / "stats.json")

    print("\nDone.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()