#!/usr/bin/env python
"""Backward-compatible wrapper for the renamed F4 embedding research step."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import load_module


_step4 = load_module("mvf_step4_embedding_research", BASE_DIR / "scripts" / "step4_embedding_research.py", REPO_ROOT)
build_parser = _step4.build_parser
main = _step4.main


if __name__ == "__main__":
    main(build_parser().parse_args())
