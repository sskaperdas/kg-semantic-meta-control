from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TripleStr = Tuple[str, str, str]
TripleId = Tuple[int, int, int]


# =========================================================
# General utilities
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# Data loading
# =========================================================

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
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def read_id_map(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid id-map format in {path} at line {line_no}: {line!r}")
            key, idx = parts
            mapping[key] = int(idx)
    return mapping


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in mapping.items()}


def triples_str_to_ids(
    triples: Sequence[TripleStr],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> List[TripleId]:
    out: List[TripleId] = []
    for h, r, t in triples:
        out.append((entity2id[h], relation2id[r], entity2id[t]))
    return out


def load_prepared_dataset(data_dir: Path) -> dict:
    train_str = read_triples_str(data_dir / "train.txt")
    valid_str = read_triples_str(data_dir / "valid.txt")
    test_str = read_triples_str(data_dir / "test.txt")

    entity2id = read_id_map(data_dir / "entity2id.txt")
    relation2id = read_id_map(data_dir / "relation2id.txt")

    train_ids = triples_str_to_ids(train_str, entity2id, relation2id)
    valid_ids = triples_str_to_ids(valid_str, entity2id, relation2id)
    test_ids = triples_str_to_ids(test_str, entity2id, relation2id)

    return {
        "train_str": train_str,
        "valid_str": valid_str,
        "test_str": test_str,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "all_ids": train_ids + valid_ids + test_ids,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "id2entity": invert_mapping(entity2id),
        "id2relation": invert_mapping(relation2id),
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
    }


# =========================================================
# Dataset wrapper
# =========================================================

class KGTripleDataset(Dataset):
    def __init__(self, triples: Sequence[TripleId]):
        self.triples = list(triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> TripleId:
        return self.triples[idx]


# =========================================================
# Filtered evaluation index
# =========================================================

@dataclass
class FilterIndex:
    hr_to_tails: Dict[Tuple[int, int], Set[int]]
    rt_to_heads: Dict[Tuple[int, int], Set[int]]


def build_filter_index(all_triples: Sequence[TripleId]) -> FilterIndex:
    hr_to_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    rt_to_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    for h, r, t in all_triples:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)

    return FilterIndex(
        hr_to_tails=hr_to_tails,
        rt_to_heads=rt_to_heads,
    )


# =========================================================
# Model definitions
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_emb.weight.data, -bound, bound)
        nn.init.uniform_(self.relation_emb.weight.data, -bound, bound)
        self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)

    def normalize_entities(self) -> None:
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)

    def score_hrt(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(heads)
        r = self.relation_emb(rels)
        t = self.entity_emb(tails)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def score_all_tails(self, heads: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(heads)          # [B, D]
        r = self.relation_emb(rels)         # [B, D]
        all_t = self.entity_emb.weight      # [E, D]
        x = h + r                           # [B, D]
        return -torch.cdist(x, all_t, p=self.p_norm)

    def score_all_heads(self, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        r = self.relation_emb(rels)         # [B, D]
        t = self.entity_emb(tails)          # [B, D]
        all_h = self.entity_emb.weight      # [E, D]
        x = t - r                           # [B, D]
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in [self.entity_re, self.entity_im, self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight.data)

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

        part1 = (rr * tr) @ hr_all.T
        part2 = (rr * ti) @ hi_all.T
        part3 = (ri * ti) @ hr_all.T
        part4 = (ri * tr) @ hi_all.T
        return part1 + part2 + part3 - part4


def build_model(model_name: str, num_entities: int, num_relations: int, embedding_dim: int) -> KGEModel:
    name = model_name.lower()
    if name == "transe":
        return TransEModel(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim)
    if name == "complex":
        return ComplExModel(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim)
    raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# Negative sampling
# =========================================================

def sample_negative_batch(
    pos_batch: torch.Tensor,
    num_entities: int,
    num_negs: int,
    device: torch.device,
    corrupt_both_sides: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create negative triples by corrupting head or tail.
    """
    batch_size = pos_batch.shape[0]

    heads = pos_batch[:, 0].unsqueeze(1).repeat(1, num_negs).reshape(-1).clone()
    rels = pos_batch[:, 1].unsqueeze(1).repeat(1, num_negs).reshape(-1).clone()
    tails = pos_batch[:, 2].unsqueeze(1).repeat(1, num_negs).reshape(-1).clone()

    if corrupt_both_sides:
        mask = torch.randint(0, 2, (batch_size * num_negs,), device=device, dtype=torch.bool)
        heads[mask] = torch.randint(0, num_entities, (int(mask.sum().item()),), device=device)
        tails[~mask] = torch.randint(0, num_entities, (int((~mask).sum().item()),), device=device)
    else:
        tails = torch.randint(0, num_entities, (batch_size * num_negs,), device=device)

    return heads, rels, tails


# =========================================================
# Training
# =========================================================

def compute_l2_regularization(model: nn.Module) -> torch.Tensor:
    reg = None
    for p in model.parameters():
        term = torch.sum(p ** 2)
        reg = term if reg is None else reg + term
    if reg is None:
        return torch.tensor(0.0)
    return reg


def train_one_epoch(
    model: KGEModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_entities: int,
    num_negs: int,
    margin: float,
    l2_weight: float,
    corrupt_both_sides: bool,
) -> Dict[str, float]:
    model.train()
    ranking_loss_fn = nn.MarginRankingLoss(margin=margin)

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        loader,
        desc="train",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            pos_h = batch[0].to(device=device, dtype=torch.long)
            pos_r = batch[1].to(device=device, dtype=torch.long)
            pos_t = batch[2].to(device=device, dtype=torch.long)
            batch_tensor = torch.stack([pos_h, pos_r, pos_t], dim=1)
        else:
            batch_tensor = torch.as_tensor(batch, dtype=torch.long, device=device)
            pos_h = batch_tensor[:, 0]
            pos_r = batch_tensor[:, 1]
            pos_t = batch_tensor[:, 2]

        neg_h, neg_r, neg_t = sample_negative_batch(
            pos_batch=batch_tensor,
            num_entities=num_entities,
            num_negs=num_negs,
            device=device,
            corrupt_both_sides=corrupt_both_sides,
        )

        pos_scores = model.score_hrt(pos_h, pos_r, pos_t)
        neg_scores = model.score_hrt(neg_h, neg_r, neg_t).view(batch_tensor.shape[0], num_negs)
        neg_scores = neg_scores.mean(dim=1)

        target = torch.ones_like(pos_scores)
        loss = ranking_loss_fn(pos_scores, neg_scores, target)

        if l2_weight > 0.0:
            l2_reg = compute_l2_regularization(model).to(device)
            loss = loss + l2_weight * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if isinstance(model, TransEModel):
            model.normalize_entities()

        batch_size = batch_tensor.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        progress.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    return {
        "loss": total_loss / max(total_examples, 1),
    }


# =========================================================
# Evaluation
# =========================================================

def compute_ranking_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {
            "mr": 0.0,
            "mrr": 0.0,
            "hits@1": 0.0,
            "hits@3": 0.0,
            "hits@10": 0.0,
        }

    n = len(ranks)
    return {
        "mr": sum(ranks) / n,
        "mrr": sum(1.0 / r for r in ranks) / n,
        "hits@1": sum(1 for r in ranks if r <= 1) / n,
        "hits@3": sum(1 for r in ranks if r <= 3) / n,
        "hits@10": sum(1 for r in ranks if r <= 10) / n,
    }


@torch.no_grad()
def evaluate_filtered_tail_prediction(
    model: KGEModel,
    triples: Sequence[TripleId],
    filter_index: FilterIndex,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    ranks: List[int] = []

    starts = range(0, len(triples), batch_size)
    progress = tqdm(
        starts,
        desc="eval-tail",
        leave=False,
        dynamic_ncols=True,
    )

    for start in progress:
        batch = triples[start:start + batch_size]
        heads = torch.tensor([h for h, _, _ in batch], dtype=torch.long, device=device)
        rels = torch.tensor([r for _, r, _ in batch], dtype=torch.long, device=device)

        scores = model.score_all_tails(heads, rels)  # [B, E]

        for i, (h, r, t_true) in enumerate(batch):
            for t in filter_index.hr_to_tails[(h, r)]:
                if t != t_true:
                    scores[i, t] = -1e9

            gold_score = scores[i, t_true].item()
            rank = int(torch.sum(scores[i] > gold_score).item()) + 1
            ranks.append(rank)

    metrics = compute_ranking_metrics(ranks)
    metrics["num_queries"] = len(triples)
    return metrics


@torch.no_grad()
def evaluate_filtered_head_prediction(
    model: KGEModel,
    triples: Sequence[TripleId],
    filter_index: FilterIndex,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    ranks: List[int] = []

    starts = range(0, len(triples), batch_size)
    progress = tqdm(
        starts,
        desc="eval-tail",
        leave=False,
        dynamic_ncols=True,
    )

    for start in progress:
        batch = triples[start:start + batch_size]
        rels = torch.tensor([r for _, r, _ in batch], dtype=torch.long, device=device)
        tails = torch.tensor([t for _, _, t in batch], dtype=torch.long, device=device)

        scores = model.score_all_heads(rels, tails)  # [B, E]

        for i, (h_true, r, t) in enumerate(batch):
            for h in filter_index.rt_to_heads[(r, t)]:
                if h != h_true:
                    scores[i, h] = -1e9

            gold_score = scores[i, h_true].item()
            rank = int(torch.sum(scores[i] > gold_score).item()) + 1
            ranks.append(rank)

    metrics = compute_ranking_metrics(ranks)
    metrics["num_queries"] = len(triples)
    return metrics


def average_metric_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = ["mr", "mrr", "hits@1", "hits@3", "hits@10"]
    out = {k: (a[k] + b[k]) / 2.0 for k in keys}
    out["num_queries"] = a.get("num_queries", 0) + b.get("num_queries", 0)
    return out


@torch.no_grad()
def evaluate_both_sides(
    model: KGEModel,
    triples: Sequence[TripleId],
    filter_index: FilterIndex,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    tail_metrics = evaluate_filtered_tail_prediction(
        model=model,
        triples=triples,
        filter_index=filter_index,
        device=device,
        batch_size=batch_size,
    )
    head_metrics = evaluate_filtered_head_prediction(
        model=model,
        triples=triples,
        filter_index=filter_index,
        device=device,
        batch_size=batch_size,
    )
    both_metrics = average_metric_dicts(head_metrics, tail_metrics)

    return {
        "head": head_metrics,
        "tail": tail_metrics,
        "both": both_metrics,
    }


# =========================================================
# Checkpointing
# =========================================================

def save_checkpoint(
    output_dir: Path,
    model: KGEModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    training_state: dict,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    ensure_dir(output_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "training_state": training_state,
        },
        output_dir / "model.pt",
    )

    save_json(config, output_dir / "config.json")
    save_json(training_state, output_dir / "training_state.json")
    save_json(entity2id, output_dir / "entity2id.json")
    save_json(relation2id, output_dir / "relation2id.json")


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline KG embedding models on prepared datasets.")

    parser.add_argument("--data-dir", required=True, help="Prepared dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be saved")

    parser.add_argument("--model", choices=["transe", "complex"], default="transe")
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-negs", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-best-on", choices=["mrr", "hits@10"], default="mrr")
    parser.add_argument("--eval-mode", choices=["tail", "both"], default="both")
    parser.add_argument("--corrupt-both-sides", action="store_true", help="Use both head and tail corruption during training")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    device = torch.device(args.device)

    data = load_prepared_dataset(data_dir)
    filter_index = build_filter_index(data["all_ids"])

    model = build_model(
        model_name=args.model,
        num_entities=data["num_entities"],
        num_relations=data["num_relations"],
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = KGTripleDataset(data["train_ids"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_negs": args.num_negs,
        "margin": args.margin,
        "l2_weight": args.l2_weight,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "device": str(device),
        "save_best_on": args.save_best_on,
        "eval_mode": args.eval_mode,
        "corrupt_both_sides": args.corrupt_both_sides,
        "num_entities": data["num_entities"],
        "num_relations": data["num_relations"],
        "train_triples": len(data["train_ids"]),
        "valid_triples": len(data["valid_ids"]),
        "test_triples": len(data["test_ids"]),
    }

    print(json.dumps(config, indent=2, ensure_ascii=False))

    history: List[dict] = []
    best_valid_score = -float("inf")
    best_epoch = -1
    best_valid_metrics = {}
    best_test_metrics = {}

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_entities=data["num_entities"],
            num_negs=args.num_negs,
            margin=args.margin,
            l2_weight=args.l2_weight,
            corrupt_both_sides=args.corrupt_both_sides,
        )

        if args.eval_mode == "tail":
            valid_metrics = {"tail": evaluate_filtered_tail_prediction(
                model=model,
                triples=data["valid_ids"],
                filter_index=filter_index,
                device=device,
                batch_size=args.eval_batch_size,
            )}
            valid_metrics["both"] = dict(valid_metrics["tail"])

            test_metrics = {"tail": evaluate_filtered_tail_prediction(
                model=model,
                triples=data["test_ids"],
                filter_index=filter_index,
                device=device,
                batch_size=args.eval_batch_size,
            )}
            test_metrics["both"] = dict(test_metrics["tail"])
        else:
            valid_metrics = evaluate_both_sides(
                model=model,
                triples=data["valid_ids"],
                filter_index=filter_index,
                device=device,
                batch_size=args.eval_batch_size,
            )
            test_metrics = evaluate_both_sides(
                model=model,
                triples=data["test_ids"],
                filter_index=filter_index,
                device=device,
                batch_size=args.eval_batch_size,
            )

        current_score = valid_metrics["both"][args.save_best_on]

        epoch_row = {
            "epoch": epoch,
            "train": train_stats,
            "valid": valid_metrics,
            "test": test_metrics,
        }
        history.append(epoch_row)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_stats['loss']:.6f} | "
            f"valid_mrr={valid_metrics['both']['mrr']:.4f} "
            f"valid_h10={valid_metrics['both']['hits@10']:.4f} | "
            f"test_mrr={test_metrics['both']['mrr']:.4f} "
            f"test_h10={test_metrics['both']['hits@10']:.4f}"
        )

        if current_score > best_valid_score:
            best_valid_score = current_score
            best_epoch = epoch
            best_valid_metrics = valid_metrics
            best_test_metrics = test_metrics

            training_state = {
                "best_epoch": best_epoch,
                "selection_metric": args.save_best_on,
                "best_valid_score": best_valid_score,
                "best_valid_metrics": best_valid_metrics,
                "best_test_metrics": best_test_metrics,
                "history": history,
            }

            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                config=config,
                training_state=training_state,
                entity2id=data["entity2id"],
                relation2id=data["relation2id"],
            )

    final_report = {
        "best_epoch": best_epoch,
        "selection_metric": args.save_best_on,
        "best_valid_metrics": best_valid_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
    }

    save_json(final_report, output_dir / "final_report.json")

    print("\nBest model summary:")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()