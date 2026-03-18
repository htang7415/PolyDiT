#!/usr/bin/env python
"""Backward-compatible wrapper for the redesigned F6 interpretability step."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import load_module


_step6 = load_module("mvf_step6_dit_interpretability", BASE_DIR / "scripts" / "step6_dit_interpretability.py", REPO_ROOT)
build_arg_parser = _step6.build_arg_parser
main = _step6.main


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
