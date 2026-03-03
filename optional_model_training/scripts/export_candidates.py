"""Export candidates from optional trained models."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TripleStr = Tuple[str, str, str]
TripleId = Tuple[int, int, int]


# =========================================================
# IO helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_triples_str(path: Path) -> List[TripleStr]:
    triples: List[TripleStr] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid triple format in {path} at line {line_no}: {line!r}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def triples_str_to_ids(
    triples: Sequence[TripleStr],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> List[TripleId]:
    out: List[TripleId] = []
    for h, r, t in triples:
        out.append((entity2id[h], relation2id[r], entity2id[t]))
    return out


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in mapping.items()}


# =========================================================
# Models
# =========================================================

class KGEModel(nn.Module):
    def score_hrt(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_all_tails(self, heads: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_all_heads(self, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransEModel(KGEModel):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, p_norm: int = 1):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

    def score_hrt(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(heads)
        r = self.relation_emb(rels)
        t = self.entity_emb(tails)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def score_all_tails(self, heads: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(heads)          # [B, D]
        r = self.relation_emb(rels)         # [B, D]
        all_t = self.entity_emb.weight      # [E, D]
        x = h + r
        return -torch.cdist(x, all_t, p=self.p_norm)

    def score_all_heads(self, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        r = self.relation_emb(rels)         # [B, D]
        t = self.entity_emb(tails)          # [B, D]
        all_h = self.entity_emb.weight      # [E, D]
        x = t - r
        return -torch.cdist(x, all_h, p=self.p_norm)


class ComplExModel(KGEModel):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)

    def score_hrt(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        hr = self.entity_re(heads)
        hi = self.entity_im(heads)
        rr = self.relation_re(rels)
        ri = self.relation_im(rels)
        tr = self.entity_re(tails)
        ti = self.entity_im(tails)

        return torch.sum(
            hr * rr * tr
            + hi * rr * ti
            + hr * ri * ti
            - hi * ri * tr,
            dim=-1,
        )

    def score_all_tails(self, heads: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        hr = self.entity_re(heads)
        hi = self.entity_im(heads)
        rr = self.relation_re(rels)
        ri = self.relation_im(rels)

        tr_all = self.entity_re.weight
        ti_all = self.entity_im.weight

        part1 = (hr * rr) @ tr_all.T
        part2 = (hi * rr) @ ti_all.T
        part3 = (hr * ri) @ ti_all.T
        part4 = (hi * ri) @ tr_all.T

        return part1 + part2 + part3 - part4

    def score_all_heads(self, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        rr = self.relation_re(rels)
        ri = self.relation_im(rels)
        tr = self.entity_re(tails)
        ti = self.entity_im(tails)

        hr_all = self.entity_re.weight
        hi_all = self.entity_im.weight

        # score(h, r, t)
        part1 = (rr * tr) @ hr_all.T
        part2 = (rr * ti) @ hi_all.T
        part3 = (ri * ti) @ hr_all.T
        part4 = (ri * tr) @ hi_all.T

        return part1 + part2 + part3 - part4


# =========================================================
# Filter indexes
# =========================================================

class FilterIndex:
    def __init__(
        self,
        hr_to_tails: Dict[Tuple[int, int], Set[int]],
        rt_to_heads: Dict[Tuple[int, int], Set[int]],
    ):
        self.hr_to_tails = hr_to_tails
        self.rt_to_heads = rt_to_heads


def build_filter_index(all_triples: Sequence[TripleId]) -> FilterIndex:
    hr_to_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    rt_to_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    for h, r, t in all_triples:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)

    return FilterIndex(hr_to_tails=hr_to_tails, rt_to_heads=rt_to_heads)


# =========================================================
# Model loading
# =========================================================

def build_model(model_name: str, num_entities: int, num_relations: int, embedding_dim: int) -> KGEModel:
    model_name = model_name.lower()
    if model_name == "transe":
        return TransEModel(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim)
    if model_name == "complex":
        return ComplExModel(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim)
    raise ValueError(f"Unsupported model: {model_name}")


def load_model_from_checkpoint(model_dir: Path, device: torch.device) -> tuple[KGEModel, dict]:
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]

    model = build_model(
        model_name=config["model"],
        num_entities=config["num_entities"],
        num_relations=config["num_relations"],
        embedding_dim=config["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


# =========================================================
# Candidate export
# =========================================================

@torch.no_grad()
def export_tail_candidates(
    model: KGEModel,
    queries: Sequence[TripleId],
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
    filter_index: FilterIndex,
    top_k: int,
    device: torch.device,
    filter_known: bool,
) -> List[dict]:
    results: List[dict] = []

    for h, r, t_true in queries:
        heads = torch.tensor([h], dtype=torch.long, device=device)
        rels = torch.tensor([r], dtype=torch.long, device=device)

        scores = model.score_all_tails(heads, rels).squeeze(0)  # [E]

        if filter_known:
            for t in filter_index.hr_to_tails[(h, r)]:
                if t != t_true:
                    scores[t] = -1e9

        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]), dim=0)

        candidates = []
        for idx, sc in zip(top_indices.tolist(), top_scores.tolist()):
            candidates.append(
                {
                    "entity_id": idx,
                    "entity": id2entity[idx],
                    "score": float(sc),
                    "is_gold": idx == t_true,
                }
            )

        results.append(
            {
                "query_type": "tail_prediction",
                "query": {
                    "head_id": h,
                    "head": id2entity[h],
                    "relation_id": r,
                    "relation": id2relation[r],
                    "gold_tail_id": t_true,
                    "gold_tail": id2entity[t_true],
                },
                "candidates": candidates,
            }
        )

    return results


@torch.no_grad()
def export_head_candidates(
    model: KGEModel,
    queries: Sequence[TripleId],
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
    filter_index: FilterIndex,
    top_k: int,
    device: torch.device,
    filter_known: bool,
) -> List[dict]:
    results: List[dict] = []

    for h_true, r, t in queries:
        rels = torch.tensor([r], dtype=torch.long, device=device)
        tails = torch.tensor([t], dtype=torch.long, device=device)

        scores = model.score_all_heads(rels, tails).squeeze(0)  # [E]

        if filter_known:
            for h in filter_index.rt_to_heads[(r, t)]:
                if h != h_true:
                    scores[h] = -1e9

        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]), dim=0)

        candidates = []
        for idx, sc in zip(top_indices.tolist(), top_scores.tolist()):
            candidates.append(
                {
                    "entity_id": idx,
                    "entity": id2entity[idx],
                    "score": float(sc),
                    "is_gold": idx == h_true,
                }
            )

        results.append(
            {
                "query_type": "head_prediction",
                "query": {
                    "gold_head_id": h_true,
                    "gold_head": id2entity[h_true],
                    "relation_id": r,
                    "relation": id2relation[r],
                    "tail_id": t,
                    "tail": id2entity[t],
                },
                "candidates": candidates,
            }
        )

    return results


def write_jsonl(records: Sequence[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-K candidates from a trained KGE model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt, config.json, entity2id.json, relation2id.json")
    parser.add_argument("--data-dir", required=True, help="Prepared data directory with train.txt, valid.txt, test.txt")
    parser.add_argument("--output-dir", required=True, help="Where to save exported candidates")

    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--direction", choices=["tail", "head", "both"], default="tail")
    parser.add_argument("--filter-known", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0, help="If > 0, export only the first N queries")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    device = torch.device(args.device)

    model, train_config = load_model_from_checkpoint(model_dir, device=device)

    entity2id = read_json(model_dir / "entity2id.json")
    relation2id = read_json(model_dir / "relation2id.json")

    # JSON keys are strings, values are ints already
    entity2id = {str(k): int(v) for k, v in entity2id.items()}
    relation2id = {str(k): int(v) for k, v in relation2id.items()}
    id2entity = invert_mapping(entity2id)
    id2relation = invert_mapping(relation2id)

    train_str = read_triples_str(data_dir / "train.txt")
    valid_str = read_triples_str(data_dir / "valid.txt")
    test_str = read_triples_str(data_dir / "test.txt")

    train_ids = triples_str_to_ids(train_str, entity2id, relation2id)
    valid_ids = triples_str_to_ids(valid_str, entity2id, relation2id)
    test_ids = triples_str_to_ids(test_str, entity2id, relation2id)
    all_ids = train_ids + valid_ids + test_ids

    filter_index = build_filter_index(all_ids)

    split_map = {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }
    query_triples = split_map[args.split]
    if args.max_queries > 0:
        query_triples = query_triples[:args.max_queries]

    run_info = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "trained_model": train_config["model"],
        "embedding_dim": train_config["embedding_dim"],
        "split": args.split,
        "top_k": args.top_k,
        "direction": args.direction,
        "filter_known": args.filter_known,
        "num_queries": len(query_triples),
        "device": str(device),
    }

    with (output_dir / "export_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(json.dumps(run_info, indent=2, ensure_ascii=False))

    if args.direction in {"tail", "both"}:
        tail_records = export_tail_candidates(
            model=model,
            queries=query_triples,
            id2entity=id2entity,
            id2relation=id2relation,
            filter_index=filter_index,
            top_k=args.top_k,
            device=device,
            filter_known=args.filter_known,
        )
        write_jsonl(tail_records, output_dir / f"{args.split}_tail_candidates.jsonl")

        with (output_dir / f"{args.split}_tail_candidates.json").open("w", encoding="utf-8") as f:
            json.dump(tail_records, f, indent=2, ensure_ascii=False)

    if args.direction in {"head", "both"}:
        head_records = export_head_candidates(
            model=model,
            queries=query_triples,
            id2entity=id2entity,
            id2relation=id2relation,
            filter_index=filter_index,
            top_k=args.top_k,
            device=device,
            filter_known=args.filter_known,
        )
        write_jsonl(head_records, output_dir / f"{args.split}_head_candidates.jsonl")

        with (output_dir / f"{args.split}_head_candidates.json").open("w", encoding="utf-8") as f:
            json.dump(head_records, f, indent=2, ensure_ascii=False)

    print("Candidate export completed.")


if __name__ == "__main__":
    main()
