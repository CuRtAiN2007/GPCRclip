#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "src/class_balanced/train_class_balanced_v2.py"
    command = [
        sys.executable,
        str(script),
        "--project",
        str(root),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
