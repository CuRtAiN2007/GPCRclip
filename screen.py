#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=root, text=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--ligand")
    parser.add_argument("--target", default="SMO_WT")
    parser.add_argument("--state", default="active")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=root / "results/results.csv")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="gpcrclip_screen_"))
    work.mkdir(parents=True, exist_ok=True)
    library_dir = work / "library"
    pdb_dir = work / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_copy = pdb_dir / args.pdb.name
    if not pdb_copy.exists():
        pdb_copy.write_bytes(args.pdb.read_bytes())

    run(
        [
            sys.executable,
            str(root / "src/scripts/prepare_library.py"),
            "--input",
            str(args.smiles),
            "--output-dir",
            str(library_dir),
            "--name",
            "library",
        ],
        root,
    )
    target_command = [
        sys.executable,
        str(root / "src/scripts/extract_pocket.py"),
        "--pdb-dir",
        str(pdb_dir),
        "--output",
        str(work / "targets.lmdb"),
        "--target",
        args.target,
        "--state",
        args.state,
    ]
    if args.ligand:
        target_command.extend(["--ligand", args.ligand])
    run(target_command, root)
    run(
        [
            sys.executable,
            str(root / "predict.py"),
            "--molecules-lmdb",
            str(library_dir / "library.lmdb"),
            "--targets-lmdb",
            str(work / "targets.lmdb"),
            "--device",
            args.device,
            "--output",
            str(args.output),
        ],
        root,
    )


if __name__ == "__main__":
    main()
