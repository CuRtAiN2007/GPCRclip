#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CLASS_BALANCED = SRC / "class_balanced"
for path in (SRC, CLASS_BALANCED, SRC / "scripts"):
    sys.path.insert(0, str(path))

from gpcr_clip.data.dictionary import Dictionary, DEFAULT_MOL_DICT, DEFAULT_POCKET_DICT
from gpcr_clip.data.lmdb_store import LMDBDataset
from gpcr_clip.data.molecules import collate_molecules, dict_to_record
from gpcr_clip.data.pockets import collate_pockets, dict_to_pocket, tokenize_pocket
from gpcr_clip.pipeline.screen import MoleculeDataset
from eval_class_balanced_v2 import load_components
from train_class_balanced_v2 import load_model


MODEL_VERSION = "gpcr_class_balanced_fold0_seed42_v1_epoch005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecules-lmdb", type=Path, required=True)
    parser.add_argument("--targets-lmdb", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "models/components",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "models/base/gpcr_official_expneg_eval_fold0_merged.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/demo_results.csv",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--background-size", type=int, default=2048)
    parser.add_argument("--class-name", default="F")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--class-f-rank", type=int, default=16)
    parser.add_argument("--class-f-alpha", type=float, default=32.0)
    return parser.parse_args()


def load_records(path: Path):
    return [dict_to_record(row) for row in LMDBDataset(path)]


def encode_molecules(
    *,
    model,
    records,
    device: torch.device,
    batch_size: int,
    class_name: str,
) -> np.ndarray:
    dataset = MoleculeDataset(records, model.backbone.mol_dict)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_molecules(
            [item[0] for item in batch]
        ),
        num_workers=0,
    )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            embedding = model.backbone.forward_mol(batch)
            embedding = model._adapt(
                embedding,
                [class_name] * embedding.size(0),
                "mol",
            )
            chunks.append(embedding.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def encode_pockets(
    *,
    model,
    pockets,
    device: torch.device,
    class_name: str,
) -> np.ndarray:
    tokenized = [
        tokenize_pocket(record, model.backbone.pocket_dict, max_seq_len=512)
        for record in pockets
    ]
    batch = collate_pockets(tokenized)
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.inference_mode():
        embedding = model.backbone.forward_pocket(batch)
        embedding = model._adapt(
            embedding,
            [class_name] * embedding.size(0),
            "pocket",
        )
    return embedding.float().cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    mol_dict = Dictionary.load(ROOT / DEFAULT_MOL_DICT)
    pocket_dict = Dictionary.load(ROOT / DEFAULT_POCKET_DICT)
    model_args = SimpleNamespace(
        base_checkpoint=args.base_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        class_f_rank=args.class_f_rank,
        class_f_alpha=args.class_f_alpha,
    )
    model = load_model(
        project=ROOT,
        args=model_args,
        mol_dict=mol_dict,
        pocket_dict=pocket_dict,
    ).to(device)
    load_info = load_components(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
    )

    pockets = [dict_to_pocket(row) for row in LMDBDataset(args.targets_lmdb)]
    records = load_records(args.molecules_lmdb)
    if not pockets or not records:
        raise RuntimeError("empty molecules or targets LMDB")
    background = records[: min(args.background_size, len(records))]

    pocket_emb = encode_pockets(
        model=model,
        pockets=pockets,
        device=device,
        class_name=args.class_name,
    )
    background_emb = encode_molecules(
        model=model,
        records=background,
        device=device,
        batch_size=args.batch_size,
        class_name=args.class_name,
    )
    molecule_emb = encode_molecules(
        model=model,
        records=records,
        device=device,
        batch_size=args.batch_size,
        class_name=args.class_name,
    )
    raw = (pocket_emb @ molecule_emb.T).astype(np.float32)
    background_raw = (pocket_emb @ background_emb.T).astype(np.float32)
    median = np.median(background_raw, axis=1, keepdims=True)
    mad = np.median(np.abs(background_raw - median), axis=1, keepdims=True)
    zscore = (
        0.6745 * (raw - median) / (mad + np.float32(1e-6))
    ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "candidate_id",
        "candidate_name",
        "smiles",
        "target",
        "state",
        "raw_similarity",
        "robust_zscore",
        "model_version",
        "notes",
    ]
    rows = []
    for pocket_index, pocket in enumerate(pockets):
        for molecule_index, record in enumerate(records):
            rows.append(
                {
                    "candidate_id": record.name,
                    "candidate_name": record.name,
                    "smiles": record.smi,
                    "target": pocket.target,
                    "state": pocket.state,
                    "raw_similarity": float(raw[pocket_index, molecule_index]),
                    "robust_zscore": float(zscore[pocket_index, molecule_index]),
                    "model_version": MODEL_VERSION,
                    "notes": (
                        "retrieval triage only; not binding affinity"
                    ),
                }
            )
    rows.sort(key=lambda row: -row["robust_zscore"])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output = dict(row)
            output["rank"] = rank
            writer.writerow(output)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        raw=raw,
        zscore=zscore,
        molecule_names=np.asarray([record.name for record in records], dtype=object),
        smiles=np.asarray([record.smi for record in records], dtype=object),
        pocket_names=np.asarray([pocket.name for pocket in pockets], dtype=object),
        targets=np.asarray([pocket.target for pocket in pockets], dtype=object),
        states=np.asarray([pocket.state for pocket in pockets], dtype=object),
    )
    summary = {
        "status": "PASS",
        "model_version": MODEL_VERSION,
        "n_molecules": len(records),
        "n_pockets": len(pockets),
        "output": str(args.output),
        "load_info": load_info,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
